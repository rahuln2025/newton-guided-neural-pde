"""
Data-driven (supervised) training pipeline for 2D PDE neural operators.

Trains a model to predict PDE solutions directly from pre-computed
ground truth data. No solver is called during training.

Usage:
    python train_pdes_supervised.py --config config_supervised.yaml

The training HDF5 must contain 'solution_tensor' — generate it using:
    utils_save_valid_dataset(..., type='train', ...)
or use compute_train_solution_stats() output.

Compare with solver-in-the-loop training (train_pdes_adaptive2.py)
by pointing both to the same test set and comparing TensorBoard logs.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before any pyplot import
from matplotlib import pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import time

from torch.cuda.amp import autocast, GradScaler

import scienceplots
plt.style.use(['science', 'grid', 'no-latex'])

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12

import csv
import matplotlib
import h5py
import yaml
import argparse

# Reuse components from the solver-in-the-loop codebase
from utils_mlflow import start_mlflow_run, log_mlflow_metrics, log_mlflow_artifact, log_mlflow_config, end_mlflow_run
from model_pdes import ImagePredictorUNet
from model_neuralop import ImagePredictorFNO
from solver import sampling_points_2d_grid, solve_pde

from utils_train_with_solver import (
    get_precision_config,
    convert_model_precision,
    create_model,
    PDEDataset,
    NormalizedMSELoss,
    resolve_run_paths,
)

from utils_plots import *


# ============================================================================
# Supervised training epoch
# ============================================================================

def train_epoch_supervised(model, epoch_ix, optimizer, loss_fn, training_loader,
                           tb_writer, scaler=None, precision_config=None):
    """
    One epoch of supervised training: forward → loss(pred, ground_truth) → backward.
    No solver calls.

    Args:
        model: Neural network model
        epoch_ix: Current epoch index (0-based)
        optimizer: Torch optimizer
        loss_fn: Loss function (NormalizedMSELoss)
        training_loader: DataLoader yielding (input_tensor, solution_tensor, sample_idx)
        tb_writer: TensorBoard SummaryWriter
        scaler: GradScaler for AMP (None if not using AMP)
        precision_config: Dict from get_precision_config()

    Returns:
        avg_loss: Average NormalizedMSE loss over the epoch
        avg_mse_loss: Average MSE loss over the epoch (for logging)
    """
    running_loss = 0.0
    running_mse_loss = 0.0
    batch_count = 0

    device = next(model.parameters()).device
    _dtype = precision_config['input_dtype'] if precision_config else torch.float32
    use_amp = scaler is not None
    mse_loss_fn = nn.MSELoss()

    model.train()
    for i, (input_tensor, solution_tensor, sample_idx) in enumerate(training_loader):
        input_tensor = input_tensor.to(device=device, dtype=_dtype)
        solution_tensor = solution_tensor.squeeze(1).to(device=device, dtype=_dtype)

        optimizer.zero_grad()

        # Forward pass
        if use_amp:
            with autocast():
                pred = model(input_tensor).squeeze(1)
                batch_loss = loss_fn(pred, solution_tensor)
                batch_mse_loss = mse_loss_fn(pred, solution_tensor)
        else:
            pred = model(input_tensor).squeeze(1)
            batch_loss = loss_fn(pred, solution_tensor)
            batch_mse_loss = mse_loss_fn(pred, solution_tensor)

        # Backward pass
        if use_amp:
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss.backward()
            optimizer.step()

        running_loss += batch_loss.item()
        running_mse_loss += batch_mse_loss.item()
        batch_count += 1

        # Per-batch TensorBoard logging
        tb_x = epoch_ix * len(training_loader) + i + 1
        tb_writer.add_scalar('Loss/train', batch_loss.item(), tb_x)
        tb_writer.add_scalar('Loss/train_mse', batch_mse_loss.item(), tb_x)

    avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
    avg_mse_loss = running_mse_loss / batch_count if batch_count > 0 else 0.0
    return avg_loss, avg_mse_loss


def val_epoch_supervised(model, epoch_ix, loss_fn, val_loader, tb_writer,
                         precision_config=None):
    """
    Validation epoch: forward → loss(pred, ground_truth). No gradients, no solver.

    Returns:
        avg_val_loss, avg_val_mse_loss
    """
    running_loss = 0.0
    running_mse_loss = 0.0
    batch_count = 0

    device = next(model.parameters()).device
    _dtype = precision_config['input_dtype'] if precision_config else torch.float32
    mse_loss_fn = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        for i, (input_tensor, solution_tensor, sample_idx) in enumerate(val_loader):
            input_tensor = input_tensor.to(device=device, dtype=_dtype)
            solution_tensor = solution_tensor.squeeze(1).to(device=device, dtype=_dtype)

            pred = model(input_tensor).squeeze(1)
            batch_loss = loss_fn(pred, solution_tensor)
            batch_mse_loss = mse_loss_fn(pred, solution_tensor)

            running_loss += batch_loss.item()
            running_mse_loss += batch_mse_loss.item()
            batch_count += 1

            # Per-batch TensorBoard logging
            total_train_batches = len(tb_writer.train_loader) if hasattr(tb_writer, 'train_loader') else 1
            tb_x = epoch_ix * total_train_batches + (epoch_ix * len(val_loader)) + i + 1
            tb_writer.add_scalar('Loss/val', batch_loss.item(), tb_x)
            tb_writer.add_scalar('Loss/val_mse', batch_mse_loss.item(), tb_x)

    avg_val_loss = running_loss / batch_count if batch_count > 0 else 0.0
    avg_val_mse_loss = running_mse_loss / batch_count if batch_count > 0 else 0.0
    return avg_val_loss, avg_val_mse_loss


# ============================================================================
# Main supervised training loop
# ============================================================================

def train_model_supervised(model, training_loader, validation_loader, writer=None,
                           config=None, optuna_trial=None):
    """
    Full supervised training loop.

    No solver pool, no adaptive maxiters — just straightforward
    loss(pred, ground_truth) training with timing for comparison.

    Args:
        optuna_trial: Optional optuna.trial.Trial object. When provided,
            reports val_loss each epoch and raises optuna.TrialPruned()
            if the trial should be stopped early.
    """
    losses = []
    val_losses = []
    epoch_times = []
    epoch_number = 0
    best_loss = float('inf')

    if config is None:
        config = {}
    config = resolve_run_paths(config)
    training_config = config['training_config']
    precision_config = get_precision_config(config)

    # --- Directories ---
    checkpoint_dir = training_config.get('checkpoint_dir')
    run_dir = training_config.get('run_dir', os.path.dirname(checkpoint_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    # Save config copy in run_dir root
    config_save_path = os.path.join(run_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to {config_save_path}")

    # --- CSV metrics logger ---
    csv_path = os.path.join(run_dir, 'metrics.csv')
    _csv_fields = ['epoch', 'train_nmse', 'val_nmse', 'train_mse', 'val_mse', 'epoch_time_s']
    _csv_file   = open(csv_path, 'w', newline='')
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields)
    _csv_writer.writeheader()
    print(f"CSV metrics logger: {csv_path}")

    # --- MLflow ---
    mlflow_run = start_mlflow_run(config)
    log_mlflow_config(mlflow_run, config, run_dir)

    # --- Log config to TensorBoard ---
    if writer is not None:
        config_text = yaml.dump(config, default_flow_style=False, sort_keys=False)
        writer.add_text('Config/YAML', f'```yaml\n{config_text}\n```', 0)

        writer.add_hparams(
            hparam_dict={
                'lr': training_config.get('learning_rate'),
                'batch_size': training_config.get('batch_size'),
                'epochs': training_config.get('epochs'),
                'precision': precision_config['precision'],
                'training_mode': 'supervised',
                'n_pts': config['data_config'].get('n_pts'),
                'train_samples': int(config['data_config'].get('total_num_samples')*config['data_config'].get('train_frac')),
                'val_samples': int(config['data_config'].get('total_num_samples')*config['data_config'].get('val_frac')),
                'in_channels': config['model_config'].get('in_channels'),
                'out_channels': config['model_config'].get('out_channels'),
                'scheduler': training_config.get('use_lr_scheduler', False),
                'scheduler_step': training_config.get('scheduler_config', {}).get('step_size', 0),
                'scheduler_gamma': training_config.get('scheduler_config', {}).get('gamma', 0),
            },
            metric_dict={'hparam/placeholder': 0},
        )
        print("Config logged to TensorBoard")

    # --- Loss, optimizer, scheduler ---
    loss_fn = NormalizedMSELoss()
    lr = training_config.get('learning_rate')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=training_config.get('weight_decay'),
    )

    device = next(model.parameters()).device
    use_amp = precision_config['use_amp']
    scaler = GradScaler() if use_amp else None

    print(f"Training precision: {precision_config['precision']} | "
          f"Model dtype: {precision_config['model_dtype']} | "
          f"AMP: {'on' if use_amp else 'off'} | Mode: supervised (no solver)")

    use_lr_scheduler = training_config.get('use_lr_scheduler', False)
    scheduler = None
    if use_lr_scheduler:
        step_size = training_config['scheduler_config']['step_size']
        gamma = training_config['scheduler_config']['gamma']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # --- Load checkpoint ---
    load_checkpoint_path = training_config.get('load_checkpoint_path')
    if load_checkpoint_path and load_checkpoint_path != 'None' and os.path.exists(load_checkpoint_path):
        checkpoint = torch.load(load_checkpoint_path, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        state_dict.pop('_metadata', None)
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if use_lr_scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_number = checkpoint.get('epoch', 0)
        losses = checkpoint.get('losses', [])
        val_losses = checkpoint.get('val_losses', [])
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resumed from checkpoint epoch {epoch_number}")

    if writer is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(f'runs/supervised_{timestamp}')

    writer.train_loader = training_loader

    epochs = training_config.get('epochs')
    total_train_time = 0.0

    # --- Inference config ---
    inference_interval = training_config.get('inference_interval', 50)
    test_sample_indices = training_config.get('test_sample_indices', [0, 1, 2, 3])
    inference_path = config['data_config'].get('train_path')
    # For periodic inference we need PDE params → use sample_type='train'
    inference_dataset = PDEDataset(inference_path, config, sample_type='train')

    # ======================== TRAINING LOOP ========================
    for epoch in range(epoch_number, epochs):
        print(f'EPOCH {epoch + 1}:')

        # --- Train ---
        t0 = time.time()
        model.train()
        avg_loss, avg_mse_loss = train_epoch_supervised(
            model, epoch, optimizer, loss_fn, training_loader,
            writer, scaler=scaler, precision_config=precision_config,
        )
        epoch_dt = time.time() - t0
        total_train_time += epoch_dt
        epoch_times.append(epoch_dt)
        losses.append(avg_loss)

        # --- Validate ---
        model.eval()
        avg_val_loss, avg_val_mse_loss = val_epoch_supervised(
            model, epoch, loss_fn, validation_loader, writer,
            precision_config=precision_config,
        )
        val_losses.append(avg_val_loss)

        # --- Print summary ---
        print(f'  NMSE  train: {avg_loss:.6f}  val: {avg_val_loss:.6f}')
        print(f'  MSE   train: {avg_mse_loss:.6f}  val: {avg_val_mse_loss:.6f}')
        print(f'  Time: {epoch_dt:.2f}s  (total: {total_train_time:.1f}s)')

        # --- Epoch-level TensorBoard ---
        writer.add_scalars('Loss/epoch', {'train': avg_loss, 'val': avg_val_loss}, epoch + 1)
        writer.add_scalars('Loss_MSE/epoch', {'train': avg_mse_loss, 'val': avg_val_mse_loss}, epoch + 1)
        writer.add_scalar('Time/epoch_seconds', epoch_dt, epoch + 1)
        writer.add_scalar('Time/total_seconds', total_train_time, epoch + 1)
        writer.flush()

        log_mlflow_metrics(mlflow_run, {
            'train_nmse': avg_loss,
            'val_nmse':   avg_val_loss,
            'train_mse':  avg_mse_loss,
            'val_mse':    avg_val_mse_loss,
            'epoch_time': epoch_dt,
        }, step=epoch + 1)

        # --- CSV ---
        _csv_writer.writerow({
            'epoch':        epoch + 1,
            'train_nmse':   avg_loss,
            'val_nmse':     avg_val_loss,
            'train_mse':    avg_mse_loss,
            'val_mse':      avg_val_mse_loss,
            'epoch_time_s': epoch_dt,
        })
        _csv_file.flush()

        if use_lr_scheduler and scheduler is not None:
            scheduler.step()

        # --- Checkpointing ---
        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_{epoch + 1}.pt')
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'val_losses': val_losses,
            'epoch_times': epoch_times,
            'best_loss': best_loss,
            'training_mode': 'supervised',
        }
        if use_lr_scheduler and scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, checkpoint_path)
            print(f'  Checkpoint saved at epoch {epoch + 1}')

        # Top-3 checkpoint management
        if not hasattr(train_model_supervised, 'top_checkpoints'):
            train_model_supervised.top_checkpoints = []
        train_model_supervised.top_checkpoints.append((avg_val_loss, epoch + 1, checkpoint_path))
        train_model_supervised.top_checkpoints.sort(key=lambda x: x[0])
        if len(train_model_supervised.top_checkpoints) > 3:
            removed = train_model_supervised.top_checkpoints.pop()
            if os.path.exists(removed[2]) and removed[1] % 10 != 0:
                os.remove(removed[2])
        if any(epoch + 1 == c[1] for c in train_model_supervised.top_checkpoints):
            torch.save(ckpt, checkpoint_path)
            print(f'  Top-3 checkpoint saved (val_loss: {avg_val_loss:.6f})')

        # Best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(ckpt, best_model_path)
            print(f'  New best model (val_loss: {best_loss:.6f})')
            log_mlflow_artifact(mlflow_run, best_model_path)

        # --- Optuna pruning hook ---
        if optuna_trial is not None:
            import optuna
            optuna_trial.report(avg_val_loss, epoch)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        # --- Periodic inference ---
        if (epoch + 1) % inference_interval == 0:
            print(f"\n{'=' * 60}")
            print(f"Running periodic inference at epoch {epoch + 1}")
            print(f"{'=' * 60}")
            inf_dir = os.path.join(run_dir, 'inference_plots')
            os.makedirs(inf_dir, exist_ok=True)
            _periodic_inference_supervised(
                model, inference_dataset, test_sample_indices,
                config, epoch + 1, inf_dir, device, writer,
            )
            log_mlflow_artifact(mlflow_run, os.path.join(inf_dir, f'epoch_{epoch + 1}'))
            print(f"{'=' * 60}\n")

    print(f"\nTraining complete. Total time: {total_train_time:.1f}s "
          f"({total_train_time / epochs:.2f}s/epoch avg)")

    _csv_file.close()
    print(f"Metrics CSV closed: {csv_path}")
    log_mlflow_artifact(mlflow_run, csv_path)

    end_mlflow_run(mlflow_run)
    return losses, val_losses, epoch_times, model


def _periodic_inference_supervised(model, dataset, sample_indices, config, epoch,
                                   plot_dir, device, writer=None):
    """Quick prediction-only inference for periodic monitoring."""
    from solver import sampling_points_2d_grid
    n_pts = config['data_config']['n_pts']
    x_pts, y_pts, xv, yv, _, _ = sampling_points_2d_grid(n_pts=n_pts)
    prec = get_precision_config(config)

    model.eval()
    epoch_dir = os.path.join(plot_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    # Clamp indices to dataset size
    max_idx = len(dataset) - 1
    sample_indices = [idx for idx in sample_indices if idx <= max_idx]

    samples = []
    for idx in sample_indices:
        item = dataset[idx]
        # dataset may return 9-tuple (train) or 3-tuple (test)
        samples.append({'input': item[0], 'idx': idx})

    if not samples:
        return

    input_batch = torch.stack(
        [torch.from_numpy(s['input']) for s in samples]
    ).to(device=device, dtype=prec['input_dtype'])

    with torch.no_grad():
        pred_batch = model(input_batch)
    pred_np = pred_batch.squeeze(1).detach().cpu().numpy()

    xv_plot, yv_plot = np.meshgrid(x_pts, y_pts, indexing='ij')
    for i, s in enumerate(samples):
        plot_2d_scatter_tensor(xv_plot, yv_plot, s['idx'], epoch_dir,
                               pred_np[i][np.newaxis, :, :],
                               savefile=f'pred_sample{s["idx"]}')

    print(f"  Plotted {len(samples)} predictions → {epoch_dir}")
    model.train()


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Supervised (data-driven) training for 2D PDE neural operators'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    return parser.parse_args()
