import math
import os
import csv
import time

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from utils_train_with_solver import (
    resolve_run_paths,
    get_precision_config,
    convert_model_precision,
    create_model,
    PDEDataset,
    parse_args,
    NormalizedMSELoss,
    _BC_SLICES,
    apply_bc_mask_torch_batch,
)
from utils_mlflow import start_mlflow_run, log_mlflow_metrics, log_mlflow_artifact, log_mlflow_config, end_mlflow_run



class FEMResidualLoss(nn.Module):
    """
    Normalized FEM residual loss on a structured 2D uniform grid.

    Discretizes the strong-form PDE residual using a 5-point FD Laplacian:

        R_ij = -(u_{i+1,j} - 2u_ij + u_{i-1,j})/h²
               -(u_{i,j+1} - 2u_ij + u_{i,j-1})/h²
               + k² * u_ij^n
               - f(x_i, y_j)

    where f = A*(mu1²+mu2²+x²+y²)*sin(mu1*pi*x)*sin(mu2*pi*y).

    Evaluated only on interior points (boundary excluded).
    Normalized per-sample by ||f_interior||² for scale invariance.

    Gradients flow through u (the network prediction) only.
    PDE parameters (k, n, A, mu1, mu2) are treated as constants.
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, u, k, n_exp, A, mu1, mu2, x_pts, y_pts):
        """
        Args:
            u:     [B, H, W] prediction tensor (BCs already enforced)
            k:     [B] tensor — diffusion coefficient
            n_exp: [B] tensor — nonlinear exponent (float, may be non-integer)
            A:     [B] tensor — forcing amplitude
            mu1:   [B] tensor — forcing frequency x
            mu2:   [B] tensor — forcing frequency y
            x_pts: 1D array/tensor of length H
            y_pts: 1D array/tensor of length W
        Returns:
            scalar residual loss
        """
        B, H, W = u.shape
        device, dtype = u.device, u.dtype

        if not isinstance(x_pts, torch.Tensor):
            x_t = torch.tensor(x_pts, device=device, dtype=dtype)
            y_t = torch.tensor(y_pts, device=device, dtype=dtype)
        else:
            x_t = x_pts.to(device=device, dtype=dtype)
            y_t = y_pts.to(device=device, dtype=dtype)

        h = float(x_t[1] - x_t[0])
        xv, yv = torch.meshgrid(x_t, y_t, indexing='ij')  # [H, W]

        A_b   = A.view(B, 1, 1)
        mu1_b = mu1.view(B, 1, 1)
        mu2_b = mu2.view(B, 1, 1)
        k_b   = k.view(B, 1, 1)
        n_b   = n_exp.view(B, 1, 1)

        # Forcing field: [B, H, W]
        f_field = A_b * (mu1_b**2 + mu2_b**2 + xv**2 + yv**2) * \
                  torch.sin(mu1_b * math.pi * xv) * torch.sin(mu2_b * math.pi * yv)

        # 5-point FD Laplacian on interior [1:-1, 1:-1]
        u_c  = u[:, 1:-1, 1:-1]
        lap  = (u[:, 2:,  1:-1] - 2*u_c + u[:, :-2, 1:-1]) / (h**2) + \
               (u[:, 1:-1, 2:] - 2*u_c + u[:, 1:-1, :-2]) / (h**2)

        # Nonlinear term k²·u^n — use |u|^n·sign(u) to handle u<0 during training
        nonlin = k_b**2 * u_c.abs().pow(n_b) * u_c.sign()

        f_int = f_field[:, 1:-1, 1:-1]

        # Residual R = -Δu + k²uⁿ - f
        R = -lap + nonlin - f_int  # [B, H-2, W-2]

        # Normalize per sample by ||f_interior||²
        R_sq = torch.sum(R**2,     dim=(1, 2))  # [B]
        f_sq = torch.sum(f_int**2, dim=(1, 2))  # [B]
        return torch.mean(R_sq / (f_sq + self.epsilon))



# ── Training functions ────────────────────────────────────────────────────────

def train_epoch_residual(model, epoch_ix, optimizer, residual_loss_fn, training_loader,
                         tb_writer, x_pts, y_pts, scaler, precision_config):
    """
    Single training epoch using only FEM residual loss — no solver calls.

    Forward pass → enforce BCs (stop grad at boundary) → FEMResidualLoss on interior.

    Args:
        model:              nn.Module
        epoch_ix:           int, 0-based epoch index
        optimizer:          torch optimizer
        residual_loss_fn:   FEMResidualLoss instance
        training_loader:    DataLoader yielding (input_tensor, alpha, f, mu1, mu2,
                            n, bcs, bc_values, sample_idxs)
        tb_writer:          SummaryWriter
        x_pts, y_pts:       1D numpy arrays of grid coordinates
        scaler:             GradScaler or None (AMP)
        precision_config:   dict from get_precision_config()

    Returns:
        avg_epoch_loss (float)
    """
    device = next(model.parameters()).device
    use_amp = scaler is not None
    _dtype  = precision_config['input_dtype']

    running_loss = 0.0
    batch_count  = 0

    model.train()
    for i, (input_tensor, alpha, f, mu1, mu2, n, bcs, bc_values, _) in enumerate(training_loader):
        input_tensor = input_tensor.to(device=device, dtype=_dtype)
        alpha        = alpha.to(device=device,     dtype=_dtype)
        f            = f.to(device=device,         dtype=_dtype)
        mu1          = mu1.to(device=device,       dtype=_dtype)
        mu2          = mu2.to(device=device,       dtype=_dtype)
        n            = n.to(device=device,         dtype=_dtype)
        bc_values    = bc_values.to(device=device, dtype=_dtype)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                pred = model(input_tensor).squeeze(1)   # [B, H, W]
        else:
            pred = model(input_tensor).squeeze(1)

        n_pts    = pred.shape[-1]
        pred_bc  = apply_bc_mask_torch_batch(pred, bcs, bc_values, n_pts)

        if use_amp:
            with autocast():
                batch_loss = residual_loss_fn(pred_bc, alpha, n, f, mu1, mu2, x_pts, y_pts)
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss = residual_loss_fn(pred_bc, alpha, n, f, mu1, mu2, x_pts, y_pts)
            batch_loss.backward()
            optimizer.step()

        running_loss += batch_loss.item()
        batch_count  += 1

        tb_x = epoch_ix * len(training_loader) + i + 1
        tb_writer.add_scalar('Loss/train_residual', batch_loss.item(), tb_x)

    return running_loss / batch_count if batch_count > 0 else 0.0


def val_epoch_supervised(model, epoch_ix, loss_fn, val_loader, tb_writer, precision_config):
    """
    Supervised validation against stored ground-truth solutions (sample_type='val').

    Returns:
        avg_val_nmse, avg_val_mse
    """
    device  = next(model.parameters()).device
    _dtype  = precision_config['input_dtype']
    mse_fn  = nn.MSELoss()

    running_nmse = 0.0
    running_mse  = 0.0
    batch_count  = 0

    model.eval()
    with torch.no_grad():
        for i, (input_tensor, solution_tensor, _) in enumerate(val_loader):
            input_tensor    = input_tensor.to(device=device, dtype=_dtype)
            solution_tensor = solution_tensor.squeeze(1).to(device=device, dtype=_dtype)

            pred = model(input_tensor).squeeze(1)

            assert pred.shape == solution_tensor.shape, \
                f"Shape mismatch: pred {pred.shape} vs target {solution_tensor.shape}"

            batch_nmse = loss_fn(pred, solution_tensor)
            batch_mse  = mse_fn(pred,  solution_tensor)

            running_nmse += batch_nmse.item()
            running_mse  += batch_mse.item()
            batch_count  += 1

            tb_x = epoch_ix * len(val_loader) + i + 1
            tb_writer.add_scalar('Loss/val_nmse', batch_nmse.item(), tb_x)
            tb_writer.add_scalar('Loss/val_mse',  batch_mse.item(),  tb_x)

    avg_nmse = running_nmse / batch_count if batch_count > 0 else 0.0
    avg_mse  = running_mse  / batch_count if batch_count > 0 else 0.0
    return avg_nmse, avg_mse


# ── Main training loop ────────────────────────────────────────────────────────

def train_model_residual(model, training_loader, validation_loader, x_pts, y_pts,
                         writer=None, config=None):
    """
    Training loop driven entirely by FEM residual loss — no solver, no multiprocessing.

    training_loader:   DataLoader with sample_type='train'  (yields PDE params)
    validation_loader: DataLoader with sample_type='val'    (yields ground truth)
    """
    if config is None:
        config = {}
    config = resolve_run_paths(config)
    training_config  = config['training_config']
    precision_config = get_precision_config(config)

    checkpoint_dir = training_config['checkpoint_dir']
    run_dir        = training_config.get('run_dir', os.path.dirname(checkpoint_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    config_save_path = os.path.join(run_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to {config_save_path}")

    mlflow_run = start_mlflow_run(config)
    log_mlflow_config(mlflow_run, config, run_dir)

    csv_path   = os.path.join(run_dir, 'metrics.csv')
    _csv_fields = ['epoch', 'train_residual', 'val_nmse', 'val_mse', 'epoch_time_s']
    _csv_file   = open(csv_path, 'w', newline='')
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields)
    _csv_writer.writeheader()
    print(f"CSV metrics: {csv_path}")

    if writer is None:
        log_path = training_config.get('log_path', os.path.join(run_dir, 'logs'))
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_dir=log_path)

    config_text = yaml.dump(config, default_flow_style=False, sort_keys=False)
    writer.add_text('Config/YAML', f'```yaml\n{config_text}\n```', 0)

    losses           = []
    val_nmses        = []
    epoch_times      = []
    total_train_time = 0.0
    best_val_loss    = float('inf')
    epoch_number     = 0

    device  = next(model.parameters()).device
    use_amp = precision_config['use_amp']
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    residual_loss_fn = FEMResidualLoss()
    nmse_loss_fn     = NormalizedMSELoss()

    lr        = training_config['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=training_config.get('weight_decay', 0.0))

    use_lr_scheduler = training_config.get('use_lr_scheduler', False)
    if use_lr_scheduler:
        sched_cfg = training_config['scheduler_config']
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sched_cfg['step_size'], gamma=sched_cfg['gamma'])

    load_checkpoint_path = training_config.get('load_checkpoint_path')
    if load_checkpoint_path and os.path.exists(load_checkpoint_path):
        ckpt = torch.load(load_checkpoint_path, weights_only=False)
        ckpt['model_state_dict'].pop('_metadata', None)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if use_lr_scheduler:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        epoch_number     = ckpt['epoch']
        losses           = ckpt.get('losses', [])
        val_nmses        = ckpt.get('val_nmses', [])
        epoch_times      = ckpt.get('epoch_times', [])
        total_train_time = sum(epoch_times)
        best_val_loss    = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from {load_checkpoint_path} at epoch {epoch_number}")

    epochs = training_config['epochs']
    print(f"Precision: {precision_config['precision']} | AMP: {use_amp}")
    print(f"Residual-only training | epochs={epochs} | lr={lr}")

    if not hasattr(train_model_residual, 'top_checkpoints'):
        train_model_residual.top_checkpoints = []

    for epoch in range(epoch_number, epochs):
        print(f'EPOCH {epoch + 1}:')
        t0 = time.time()

        avg_train_loss = train_epoch_residual(
            model, epoch, optimizer, residual_loss_fn,
            training_loader, writer, x_pts, y_pts, scaler, precision_config)
        losses.append(avg_train_loss)

        avg_val_nmse, avg_val_mse = val_epoch_supervised(
            model, epoch, nmse_loss_fn, validation_loader, writer, precision_config)
        val_nmses.append(avg_val_nmse)

        epoch_dt = time.time() - t0
        total_train_time += epoch_dt
        epoch_times.append(epoch_dt)

        print(f'  train_residual={avg_train_loss:.6f}  '
              f'val_nmse={avg_val_nmse:.6f}  val_mse={avg_val_mse:.6f}  '
              f't={epoch_dt:.2f}s')

        writer.add_scalars('Loss/epoch', {'train_residual': avg_train_loss,
                                          'val_nmse': avg_val_nmse}, epoch + 1)
        writer.add_scalar('Time/epoch_seconds', epoch_dt, epoch + 1)
        writer.add_scalar('Time/total_seconds', total_train_time, epoch + 1)
        writer.flush()

        log_mlflow_metrics(mlflow_run, {
            'train_residual': avg_train_loss,
            'val_nmse':       avg_val_nmse,
            'val_mse':        avg_val_mse,
            'epoch_time':     epoch_dt,
        }, step=epoch + 1)

        _csv_writer.writerow({
            'epoch':          epoch + 1,
            'train_residual': avg_train_loss,
            'val_nmse':       avg_val_nmse,
            'val_mse':        avg_val_mse,
            'epoch_time_s':   epoch_dt,
        })
        _csv_file.flush()

        if use_lr_scheduler:
            scheduler.step()

        checkpoint = {
            'epoch':                epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses':               losses,
            'val_nmses':            val_nmses,
            'epoch_times':          epoch_times,
            'best_val_loss':        best_val_loss,
        }
        if use_lr_scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'ckpt_{epoch + 1}.pt'))

        # top-3 by val_nmse
        train_model_residual.top_checkpoints.append((avg_val_nmse, epoch + 1))
        train_model_residual.top_checkpoints.sort(key=lambda x: x[0])
        if len(train_model_residual.top_checkpoints) > 3:
            train_model_residual.top_checkpoints = train_model_residual.top_checkpoints[:3]
        if any(epoch + 1 == e for _, e in train_model_residual.top_checkpoints):
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'top3_ckpt_epoch{epoch + 1}.pt'))

        if avg_val_nmse < best_val_loss:
            best_val_loss = avg_val_nmse
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f'  New best model saved (val_nmse={best_val_loss:.6f})')

    print(f"\nTraining complete. Total: {total_train_time:.1f}s "
          f"({total_train_time / max(epochs, 1):.2f}s/epoch avg)")

    _csv_file.close()
    log_mlflow_artifact(mlflow_run, csv_path)
    end_mlflow_run(mlflow_run)

    return losses, val_nmses, epoch_times, model

