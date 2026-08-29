import torch
import torch.optim as optim
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
import torch.multiprocessing as mp

import concurrent.futures
from functools import partial
import atexit
import traceback

import scienceplots
plt.style.use(['science',  'grid', 'no-latex'])

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300   
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12

# imports
from ngsolve import *
# from ngsolve.webgui import Draw
from ngsolve import *
# from ngsolve.webgui import *
from netgen.occ import *



from model_pdes import *
from model_neuralop import ImagePredictorFNO
from solver import *
from utils_plots import *
from utils_mlflow import start_mlflow_run, log_mlflow_metrics, log_mlflow_artifact, log_mlflow_config, end_mlflow_run

import csv
import h5py
import yaml
import argparse
from torchinfo import summary


def resolve_run_paths(config):
    """
    Populate checkpoint_dir, log_path, log_name, run_id, and mlflow fields
    from a single (run_root, run_name) pair when those fields are absent.

    Config schema (minimal):
        run_root: ./runs          # parent directory for all runs
        run_name: 256_UNet2D_dd   # experiment identifier (also MLflow experiment_name)

    Derived layout (one unique subdirectory per invocation):
        <run_root>/<run_name>/<run_id>/checkpoints/  → training_config.checkpoint_dir
        <run_root>/<run_name>/<run_id>/logs/         → training_config.log_path
        <run_id>                                      → training_config.log_name
        <run_name>                                    → mlflow_config.experiment_name
        <run_id>                                      → mlflow_config.run_name

    run_id = <run_name>_<YYYYMMDD_HHMM>  — unique per training invocation.
    Stored in config['run_id'] so all subsystems share the same id.

    Existing explicit values in the config are never overwritten.
    """
    from datetime import datetime as _dt
    run_name = config.get('run_name')
    run_root = config.get('run_root', './runs')
    if not run_name:
        return config

    tc = config.setdefault('training_config', {})
    mc = config.setdefault('mlflow_config', {})

    # Generate a stable unique id once; reuse if already set (e.g. resume)
    if not config.get('run_id'):
        config['run_id'] = f"{run_name}_{_dt.now().strftime('%Y%m%d_%H%M')}"
    run_id = config['run_id']

    # experiment_dir/<run_id>/  ← unique per invocation
    run_dir = os.path.join(run_root, run_name, run_id)

    if not tc.get('checkpoint_dir'):
        tc['checkpoint_dir'] = os.path.join(run_dir, 'checkpoints')
    if not tc.get('log_path'):
        tc['log_path'] = os.path.join(run_dir, 'logs')
    if not tc.get('log_name'):
        tc['log_name'] = run_id
    if not mc.get('experiment_name'):
        mc['experiment_name'] = run_name   # groups all runs of same experiment
    if not mc.get('run_name'):
        mc['run_name'] = run_id            # unique label inside the experiment

    # Store run_dir for callers that need the root (CSV, plots, etc.)
    tc.setdefault('run_dir', run_dir)

    return config


# Handle different precisions

def get_precision_config(config):
    """
    Centralized precision configuration for the training pipeline.

    Reads 'precision' from config['training_config'] and returns a dict
    controlling dtypes throughout training. The solver always runs in fp64.

    Options:
        'mixed'   - AMP: fp32 model weights, fp16 forward via autocast, GradScaler
        'float32' - Standard fp32 throughout (default)
        'float64' - Full double precision training
        'float16' - Pure fp16 (no loss scaling — use with caution)

    Backward compatible: falls back to 'use_amp' if 'precision' is not set.
    """
    training_config = config.get('training_config', {})
    precision = training_config.get('precision', None)

    # Backward compatibility: if 'precision' not set, check legacy 'use_amp'
    if precision is None:
        use_amp = training_config.get('use_amp', False)
        if use_amp:
            precision = 'mixed'
            print("WARNING: 'use_amp' is deprecated. Use 'precision: mixed' in training_config instead.")
        else:
            precision = 'float32'

    if precision == 'mixed':
        return {
            'precision': 'mixed',
            'model_dtype': torch.float32,     # model stays fp32; autocast handles fp16 internally
            'input_dtype': torch.float32,     # inputs fp32; autocast converts during forward
            'numpy_dtype': np.float32,
            'use_amp': True,
            'solver_output_torch_dtype': torch.float32,
        }
    elif precision == 'float64':
        return {
            'precision': 'float64',
            'model_dtype': torch.float64,
            'input_dtype': torch.float64,
            'numpy_dtype': np.float64,
            'use_amp': False,
            'solver_output_torch_dtype': torch.float64,
        }
    elif precision == 'float16':
        return {
            'precision': 'float16',
            'model_dtype': torch.float16,
            'input_dtype': torch.float16,
            'numpy_dtype': np.float32,        # solver I/O stays fp32 minimum
            'use_amp': False,
            'solver_output_torch_dtype': torch.float32,
        }
    else:  # default float32
        return {
            'precision': 'float32',
            'model_dtype': torch.float32,
            'input_dtype': torch.float32,
            'numpy_dtype': np.float32,
            'use_amp': False,
            'solver_output_torch_dtype': torch.float32,
        }


def convert_model_precision(model, target_real_dtype):
    """
    Convert all model parameters and buffers to the target precision,
    correctly handling complex-valued tensors (e.g. FNO spectral weights).

    PyTorch's .to(dtype=float64) only converts real floating-point tensors;
    complex tensors are left unchanged. This function maps both:
        float32 → float64   AND   complex64 → complex128
    """
    _complex_map = {
        torch.float16: torch.complex32,
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }
    target_complex_dtype = _complex_map.get(target_real_dtype)

    def convert(t):
        if t.is_floating_point():
            return t.to(dtype=target_real_dtype)
        elif t.is_complex() and target_complex_dtype is not None:
            return t.to(dtype=target_complex_dtype)
        return t  # leave int/bool tensors untouched

    return model._apply(convert)


def create_model(config):
    """Factory function to create model based on config['model_config']['model_type']."""
    model_type = config.get('model_config', {}).get('model_type', 'unet')
    
    if model_type == 'unet':
        print(f"Creating UNet model")
        return ImagePredictorUNet(config=config)
    elif model_type == 'fno':
        print(f"Creating FNO model")
        return ImagePredictorFNO(config=config)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'unet' or 'fno'.")



class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, config, sample_type = 'train'):
        self.file_path = file_path
        if config is None:
            config = {}

        data_config = config.get('data_config')
        self.sample_type = sample_type
        self.alpha_range = data_config.get('k_range')
        self.n_set = data_config.get('n_set')
        self.f_range = data_config.get('A_range')
        self.mu1_range = data_config.get('mu1_range')
        self.mu2_range = data_config.get('mu2_range')
        
        with h5py.File(file_path, 'r') as f:
            self.num_samples = f['k'].shape[0]

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            input_tensor = f['input_tensor'][idx]
            sample_idx = f['sample_index'][idx]
            #print(input_tensor.shape)
            if self.sample_type == 'train':
                # unscaled PDE params
                alpha = f['k'][idx]
                f_param = f['A'][idx]
                mu1 = f['mu1'][idx]
                mu2 = f['mu2'][idx]
                n = f['n'][idx]
                dirichlet_bcs = f['boundary_conditions'][idx]
                dirichlet_value = f['dirichlet_value'][idx]

                
                return input_tensor, alpha, f_param, mu1, mu2, n, dirichlet_bcs, dirichlet_value, sample_idx
            else:
                solution_tensor = f['solution_tensor'][idx]
                return input_tensor, solution_tensor, sample_idx

def parse_args():
    parser = argparse.ArgumentParser(description='Train PINN model for 3D elasticity problem using YAML config')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (default: 0)')
    return parser.parse_args()

class NormalizedMSELoss(nn.Module):
    """
    Normalized Mean Squared Error Loss - Safe implementation without view/flatten.
    
    L_NMSE = (1/|B|) * sum_over_batch( ||prediction - target||^2 / (||target||^2 + epsilon) )
    
    Args:
        epsilon: Small constant for numerical stability (default: 1e-8)
        reduction: 'mean' or 'sum' over batch dimension (default: 'mean')
    """
    def __init__(self, epsilon=1e-8, reduction='mean'):
        super(NormalizedMSELoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, prediction, target):
        """
        Args:
            prediction: Tensor of shape [B, H, W] or [B, C, H, W]
            target: Tensor of shape [B, H, W] or [B, C, H, W]
        
        Returns:
            Normalized MSE loss (scalar if reduction='mean')
        """
        # Compute squared difference
        squared_diff = (prediction - target) ** 2  # [B, H, W] or [B, C, H, W]
        squared_target = target ** 2                # [B, H, W] or [B, C, H, W]
        
        # Determine which dimensions to sum over (all except batch)
        if prediction.ndim == 3:  # [B, H, W]
            sum_dims = (1, 2)
        elif prediction.ndim == 4:  # [B, C, H, W]
            sum_dims = (1, 2, 3)
        else:
            raise ValueError(f"Expected 3D [B, H, W] or 4D [B, C, H, W] tensors, got shape {prediction.shape}")
        
        # Compute L2 norms squared by summing over spatial (and channel) dimensions
        # Result: [B]
        squared_error = torch.sum(squared_diff, dim=sum_dims)      # [B] (using torch.mean(squared_diff, dim=sum_dims) will mean over all pixels to add scale invariance per pixel, and the loss may be less sensitive to grid resolution since it averages over spatial dimensions)
        target_norm_squared = torch.sum(squared_target, dim=sum_dims)  # [B]
        
        # Normalize per sample: [B]
        normalized_loss = squared_error / (target_norm_squared + self.epsilon)
        
        # Reduction over batch
        if self.reduction == 'mean':
            return torch.mean(normalized_loss)
        elif self.reduction == 'sum':
            return torch.sum(normalized_loss)
        else:  # 'none'
            return normalized_loss



# ── Boundary masking ─────────────────────────────────────────────────────────
# Index convention (matches solver.py shape definition):
#   1 = left   (x=0  → grid i=0)
#   2 = right  (x=1  → grid i=n_pts-1)
#   3 = bottom (y=0  → grid j=0)
#   4 = top    (y=1  → grid j=n_pts-1)
_BC_SLICES = {
    1: (0,    slice(None)),
    2: (-1,   slice(None)),
    3: (slice(None), 0),
    4: (slice(None), -1),
}

def apply_bc_mask_np(pred_2d, bcs_1d, bc_val, n_pts):
    """
    Set boundary pixels of a [n_pts, n_pts] numpy array to bc_val.
    Used to provide a clean warm-start to the solver (interior-only prediction).
    """
    out = pred_2d.copy()
    for idx in bcs_1d:
        idx = int(idx)
        if idx in _BC_SLICES:
            out[_BC_SLICES[idx]] = float(bc_val)
    return out


def apply_bc_mask_torch_batch(pred_batch, bcs_batch, bc_values_batch, n_pts):
    """
    Zero gradient contribution of boundary pixels in a [B, H, W] prediction tensor.

    Multiplies boundary pixels by 0 (stopping gradient) and adds the known BC
    value (detached). Interior pixels are unchanged — full gradient flows through.

    Args:
        pred_batch:      [B, n_pts, n_pts] float tensor with grad
        bcs_batch:       [B, N] int tensor of boundary indices (0 = inactive)
        bc_values_batch: [B] float tensor of Dirichlet values
        n_pts:           grid resolution

    Returns:
        [B, n_pts, n_pts] tensor — interior pixels unchanged, boundary = bc_val (no grad)
    """
    device = pred_batch.device
    dtype  = pred_batch.dtype
    B = pred_batch.shape[0]

    interior_mask = torch.ones(B, n_pts, n_pts, device=device, dtype=dtype)
    bc_field      = torch.zeros(B, n_pts, n_pts, device=device, dtype=dtype)

    for b in range(B):
        bc_val = float(bc_values_batch[b])
        for idx in bcs_batch[b]:
            idx = int(idx)
            if idx in _BC_SLICES:
                interior_mask[b][_BC_SLICES[idx]] = 0.0
                bc_field[b][_BC_SLICES[idx]]      = bc_val

    # pred * interior_mask  → 0 at boundary (no grad), identity at interior
    # + bc_field            → restores known BC value at boundary (detached)
    return pred_batch * interior_mask + bc_field
# ─────────────────────────────────────────────────────────────────────────────


# Utils for checking model training progress

def periodic_inference(model, test_dataset, sample_indices, x_pts, y_pts, config, epoch, plot_dir, device, writer=None):
    """
    Perform inference on selected test samples and log to tensorboard.
    Only plots predictions, no solver calls or refinement.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        sample_indices: List of sample indices to evaluate
        x_pts, y_pts: Grid points
        config: Configuration dictionary
        epoch: Current epoch number
        plot_dir: Directory to save plots
        device: torch device
        writer: TensorBoard writer (optional)
    
    Returns:
        Dictionary with inference metrics
    """
    model.eval()
    epoch_dir = os.path.join(plot_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Batch load samples
    samples_data = []
    for idx in sample_indices:
        input_tensor, alpha, f, mu1, mu2, n, bcs, bc_values, sample_idx = test_dataset[idx]
        samples_data.append({
            'input_tensor': input_tensor,
            'alpha': alpha,
            'f': f,
            'mu1': mu1,
            'mu2': mu2,
            'n': n,
            'bcs': bcs,
            'bc_values': bc_values,
            'sample_idx': sample_idx,
            'idx': idx
        })
    
    # Batch inference
    prec = get_precision_config(config)
    input_batch = torch.stack([torch.from_numpy(s['input_tensor']) for s in samples_data]).to(device=device, dtype=prec['input_dtype'])
    
    with torch.no_grad():
        pred_batch = model(input_batch)
    
    pred_batch_np = pred_batch.squeeze(1).detach().cpu().numpy()
    
    xv, yv = np.meshgrid(x_pts, y_pts, indexing='ij')
    
    # Process and plot samples (predictions only)
    for i, sample in enumerate(samples_data):
        idx = sample['idx']
        sample_idx = sample['sample_idx']
        pred = pred_batch_np[i]
        
        # Plot prediction only
        plot_2d_scatter_tensor(xv, yv, sample_idx, epoch_dir, 
                              pred[np.newaxis, :, :], 
                              savefile=f'pred_sample{sample_idx}')
        
        # Optionally log prediction statistics to tensorboard
        if writer is not None:
            pred_mean = np.mean(pred)
            pred_std = np.std(pred)
            pred_min = np.min(pred)
            pred_max = np.max(pred)
            
            writer.add_scalar(f'Inference/pred_mean_sample_{sample_idx}', pred_mean, epoch)
            writer.add_scalar(f'Inference/pred_std_sample_{sample_idx}', pred_std, epoch)
            writer.add_scalar(f'Inference/pred_range_sample_{sample_idx}', pred_max - pred_min, epoch)
    
    print(f"\n[Epoch {epoch}] Inference Summary:")
    print(f"  Plotted predictions for {len(sample_indices)} samples")
    print(f"  Plots saved to: {epoch_dir}")
    
    model.train()  # Switch back to training mode
    
    return {
        'num_samples': len(sample_indices),
        'plot_dir': epoch_dir
    }



## TRAIN FUNCTIONS

# Newton solve for single sample
def process_single_sample(args_dict):
    """
    Wrapper function for solve_pde_mp that unpacks arguments from dictionary.
    This function is designed to work with ProcessPoolExecutor.submit()
    
    Args:
        args_dict: Dictionary containing all solver arguments
        
    Returns:
        Dictionary with solver results or error information
    """
    try:
        u_sol, iters, _, cg_iters, l2_dist, _ = solve_pde(
            args_dict['u_init'],
            args_dict['k'],
            args_dict['n'],
            args_dict['A'],
            args_dict['mu1'],
            args_dict['mu2'],
            args_dict['dirichlet_bcs'],
            args_dict['dirichlet_value'],
            x_pts=args_dict['x_pts'],
            y_pts=args_dict['y_pts'],
            maxiters=args_dict['maxiters'],
            verbose=args_dict.get('verbose', False),
            linear_solver=args_dict.get('linear_solver'),
            cg_maxiter=args_dict.get('cg_maxiter'),
            cg_tol=args_dict.get('cg_tol'),
            dampfactor=args_dict.get('dampfactor', 1.0),
        )

        conv_iters = None
        if args_dict.get('track_convergence', False):
            _, conv_iters, _, _, _, _ = solve_pde(
                args_dict['u_init'],
                args_dict['k'],
                args_dict['n'],
                args_dict['A'],
                args_dict['mu1'],
                args_dict['mu2'],
                args_dict['dirichlet_bcs'],
                args_dict['dirichlet_value'],
                x_pts=args_dict['x_pts'],
                y_pts=args_dict['y_pts'],
                maxiters=args_dict['convergence_maxiters'],
                verbose=False,
                linear_solver=args_dict.get('linear_solver'),
                cg_maxiter=args_dict.get('cg_maxiter'),
                cg_tol=args_dict.get('cg_tol'),
                dampfactor=args_dict.get('dampfactor', 1.0),
            )

        return {
            'status': 'success',
            'batch_idx': args_dict['batch_idx'],
            'sample_idx': args_dict['sample_idx'],
            'u_sol': u_sol,
            'iters': iters,
            'cg_iters': cg_iters,
            'l2_dist': l2_dist,
            'conv_iters': conv_iters,
        }
    
    except Exception as e:
        print(f"Worker {os.getpid()}: Error in solver for sample {args_dict.get('sample_idx', 'unknown')}: {str(e)}")
        return {
            'status': 'error',
            'batch_idx': args_dict['batch_idx'],
            'sample_idx': args_dict.get('sample_idx', -1),
            'error': str(e)
        }        



def train_epoch_mp(model, epoch_ix, optimizer, loss_fn, training_loader, tb_writer, x_pts, y_pts, maxiters, scaler, solver_pool, num_processes=2, delta_M_high=1.0e-3, delta_M_low=1.0e-4, precision_config=None, solver_config=None, residual_config=None):
    running_loss = 0. # tracks the loss over the whole epoch (gets contribution from all batches)
    running_mse_loss = 0.
    running_residual_loss = 0.
    batch_count = 0
    total_epoch_iters = 0
    total_epoch_cg_iters = 0
    total_epoch_l2_dist = 0.0 # Track batch l2 dist between u_init and u_sol for solver across epoch
    total_epoch_conv_iters = 0

    device = next(model.parameters()).device
    use_amp = scaler is not None

    # CG solver config
    if solver_config is None:
        solver_config = {}
    linear_solver = solver_config.get('linear_solver')
    cg_maxiter = int(solver_config.get('cg_maxiter'))
    cg_tol = float(solver_config.get('cg_tol'))
    track_convergence = bool(solver_config.get('track_convergence', False))
    convergence_maxiters = int(solver_config.get('convergence_maxiters', maxiters))
    newton_dampfactor = float(solver_config.get('newton_dampfactor', 1.0))

    # Create MSE loss function for logging
    mse_loss_fn = nn.MSELoss()

    # Residual loss config
    if residual_config is None:
        residual_config = {}
    use_residual_loss = bool(residual_config.get('use_residual_loss'))
    use_solver_loss   = bool(residual_config.get('use_solver_loss'))   # default True for backward compat
    lambda_residual   = float(residual_config.get('lambda_residual'))
    if use_residual_loss:
        from utils_train_residual import FEMResidualLoss  # lazy to avoid circular import
    residual_loss_fn  = FEMResidualLoss() if use_residual_loss else None

    # Loop over each batch in an epoch
    for i, (input_tensor, alpha, f, mu1, mu2, n, bcs, bc_values, sample_idxs) in enumerate(training_loader):
        input_tensor = input_tensor.to(device=device, dtype=precision_config['input_dtype'])
        
        optimizer.zero_grad()

        # Forward pass
        if use_amp:
            with autocast():
                pred = model(input_tensor)
        else:
            pred = model(input_tensor)
        
        # Ensure at least fp32 for solver input (fixes fp16 -> solver precision loss with AMP)
        pred_numpy = pred.squeeze(1).detach().to(dtype=precision_config['solver_output_torch_dtype']).cpu().numpy()

        batch_size = input_tensor.shape[0]
        
        # Prepare arguments for parallel solver calls
        solver_args = [] # list of solver_arg dictionaries for each sample in batch
        # loop over each sample in batch
        n_pts = pred_numpy.shape[-1]
        for batch_idx in range(batch_size):
            _bcs_1d = bcs[batch_idx].tolist() if isinstance(bcs[batch_idx], torch.Tensor) else bcs[batch_idx]
            _bc_val = float(bc_values[batch_idx])
            # enforce known BCs on boundary pixels before warm-starting the solver
            _pred_init = apply_bc_mask_np(pred_numpy[batch_idx], _bcs_1d, _bc_val, n_pts)
            solver_args.append({
                'u_init': _pred_init,
                'k': float(alpha[batch_idx]),
                'n': int(n[batch_idx]),
                'A': float(f[batch_idx]),
                'mu1': float(mu1[batch_idx]),
                'mu2': float(mu2[batch_idx]),
                'dirichlet_bcs': _bcs_1d,
                'dirichlet_value': _bc_val,
                'x_pts': x_pts,
                'y_pts': y_pts,
                'maxiters': maxiters,
                'verbose': False,
                'sample_idx': int(sample_idxs[batch_idx]),
                'batch_idx': batch_idx,
                'linear_solver': linear_solver,
                'cg_maxiter': cg_maxiter,
                'cg_tol': cg_tol,
                'track_convergence': track_convergence,
                'convergence_maxiters': convergence_maxiters,
                'dampfactor': newton_dampfactor,
            })
        
        # Submit all solver tasks to the pool
        try:
            # Submit tasks and collect futures
            futures = []
            for args_dict in solver_args:
                future = solver_pool.submit(process_single_sample, args_dict) # future contains the outputs from process_single_sample, i.e. the u_sol
                futures.append((future, args_dict['sample_idx'], args_dict['batch_idx']))

            # here the pred and solver outputs are already available
            
            # Collect results
            batch_loss = 0.
            batch_total_iters = 0
            batch_total_cg_iters = 0
            batch_total_l2_dist = 0.0 # Track l2 distance for the batch
            batch_total_conv_iters = 0
            
            u_sols = [None] * batch_size
            iters_list = [None] * batch_size
            l2_dist_list = [None] * batch_size
            
            # loop over each sample output in batch
            for future, sample_idx, batch_idx in futures:
                try:
                    # Unpack dictionary result instead of tuple
                    result = future.result(timeout=60)
                    
                    if result['status'] == 'success':
                        u_sols[batch_idx] = result['u_sol']
                        iters = result['iters']
                        cg_iters_sample = result['cg_iters']
                        l2_dist = result['l2_dist']

                        iters_list[batch_idx] = iters
                        l2_dist_list[batch_idx] = l2_dist
                        
                        # cg_iters is a list (one entry per Newton step) or 0 for direct solver
                        cg_sum = int(sum(cg_iters_sample)) if isinstance(cg_iters_sample, list) else int(cg_iters_sample)

                        # Log per-sample iterations
                        tb_z = epoch_ix * len(training_loader) + i + 1
                        if sample_idx == 1:
                            tb_writer.add_scalar('Sample_1_iters', iters, tb_z)
                            tb_writer.add_scalar('Sample_1_l2_dist', l2_dist, tb_z)
                        if sample_idx == 28:
                            tb_writer.add_scalar('Sample_28_iters', iters, tb_z)
                            tb_writer.add_scalar('Sample_28_l2_dist', l2_dist, tb_z)
                        
                        batch_total_iters += iters
                        batch_total_cg_iters += cg_sum
                        batch_total_l2_dist += l2_dist
                        if result.get('conv_iters') is not None:
                            batch_total_conv_iters += result['conv_iters']
                    
                    
                    else: # solver has failed
                        # Handle solver failure
                        print(f"Solver failed for sample {sample_idx}: {result.get('error', 'Unknown error')}")
                        u_sols[batch_idx] = pred_numpy[batch_idx]
                        iters_list[batch_idx] = maxiters
                        l2_dist_list[batch_idx] = 0.0  # Assume no error if failed
                        batch_total_iters += maxiters

                # multiprocessing failure or solver error in pool    
                except concurrent.futures.TimeoutError:
                    print(f"Solver timeout for sample {sample_idx} at batch {batch_idx}")
                    u_sols[batch_idx] = pred_numpy[batch_idx]
                    iters_list[batch_idx] = maxiters
                    l2_dist_list[batch_idx] = 0.0
                    batch_total_iters += maxiters
                except Exception as e:
                    print(f"Solver error for sample {sample_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    u_sols[batch_idx] = pred_numpy[batch_idx]
                    iters_list[batch_idx] = maxiters
                    l2_dist_list[batch_idx] = 0.0
                    batch_total_iters += maxiters
            
            # all solver outputs, iterations and l2 distances are collected for the batch 
            # all predictions available for the batch

            # Compute losses for all sample
            valid_samples = []
            valid_preds   = []
            valid_bcs     = []
            valid_bc_vals = []
            valid_alpha   = []
            valid_f       = []
            valid_mu1     = []
            valid_mu2     = []
            valid_n       = []

            # collect solver outputs as torch tensors
            for batch_idx in range(batch_size):
                if u_sols[batch_idx] is not None:
                    u_sol_tensor = torch.tensor(u_sols[batch_idx], dtype=precision_config['solver_output_torch_dtype'], device=device)
                    if u_sol_tensor.dim() == 2:
                        u_sol_tensor = u_sol_tensor.unsqueeze(0)  # [1, H, W]
                    valid_samples.append(u_sol_tensor)
                    valid_preds.append(pred[batch_idx:batch_idx+1].squeeze(1))  # [1, H, W]
                    valid_bcs.append(bcs[batch_idx].unsqueeze(0) if isinstance(bcs[batch_idx], torch.Tensor) else torch.tensor([bcs[batch_idx]]))
                    valid_bc_vals.append(float(bc_values[batch_idx]))
                    valid_alpha.append(float(alpha[batch_idx]))
                    valid_f.append(float(f[batch_idx]))
                    valid_mu1.append(float(mu1[batch_idx]))
                    valid_mu2.append(float(mu2[batch_idx]))
                    valid_n.append(float(n[batch_idx]))

            # ---------------------- compute batch loss ----------------------------
            if len(valid_samples) > 0:
                u_sol_batch = torch.cat(valid_samples, dim=0)  # [N, H, W]
                pred_batch  = torch.cat(valid_preds,   dim=0)  # [N, H, W]

                # mask boundary pixels: no gradient flows through known BC locations
                _bcs_t    = torch.cat(valid_bcs, dim=0).to(device=device)  # [N, K]
                _bc_vals_t = torch.tensor(valid_bc_vals, device=device, dtype=pred_batch.dtype)  # [N]
                pred_batch = apply_bc_mask_torch_batch(pred_batch, _bcs_t, _bc_vals_t, pred_batch.shape[-1])

                if use_amp:
                    with autocast():
                        batch_loss     = loss_fn(pred_batch, u_sol_batch)     if use_solver_loss else torch.tensor(0.0, device=device)
                        batch_mse_loss = mse_loss_fn(pred_batch, u_sol_batch) if use_solver_loss else torch.tensor(0.0, device=device)
                else:
                    batch_loss     = loss_fn(pred_batch, u_sol_batch)     if use_solver_loss else torch.tensor(0.0, device=device)
                    batch_mse_loss = mse_loss_fn(pred_batch, u_sol_batch) if use_solver_loss else torch.tensor(0.0, device=device)

                # Scale loss to account for failed samples
                # If we had 32 samples but only 30 valid, we need to weight appropriately
                num_valid = len(valid_samples)
                batch_loss = batch_loss #* (num_valid / batch_size)  # batch loss is now a scalar here

                # FEM residual loss (no solver, fully differentiable)
                batch_residual_loss = torch.tensor(0.0, device=device)
                if use_residual_loss:
                    _k_t   = torch.tensor(valid_alpha, device=device, dtype=pred_batch.dtype)
                    _A_t   = torch.tensor(valid_f,     device=device, dtype=pred_batch.dtype)
                    _mu1_t = torch.tensor(valid_mu1,   device=device, dtype=pred_batch.dtype)
                    _mu2_t = torch.tensor(valid_mu2,   device=device, dtype=pred_batch.dtype)
                    _n_t   = torch.tensor(valid_n,     device=device, dtype=pred_batch.dtype)
                    if use_amp:
                        with autocast():
                            batch_residual_loss = residual_loss_fn(
                                pred_batch, _k_t, _n_t, _A_t, _mu1_t, _mu2_t, x_pts, y_pts)
                    else:
                        batch_residual_loss = residual_loss_fn(
                            pred_batch, _k_t, _n_t, _A_t, _mu1_t, _mu2_t, x_pts, y_pts)
                    batch_loss = batch_loss + lambda_residual * batch_residual_loss
                
            else:
                # if no valid samples, then batch loss zero
                batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
                batch_mse_loss = torch.tensor(0.0, device=device)
        
        # multiprocessing implementation failed
        # rolling back to sequential implementation
        except Exception as e:
            print(f"Batch processing error: {str(e)}")
            traceback.print_exc()
        
        # NormalizedMSELoss already computes mean over batch dimension
        # average batch iters and l2 dist
        avg_batch_iters = batch_total_iters / batch_size
        avg_batch_cg_per_newton = batch_total_cg_iters / max(batch_total_iters, 1)
        avg_batch_l2_dist = batch_total_l2_dist / batch_size
        avg_batch_conv_iters = batch_total_conv_iters / batch_size if track_convergence else 0.0

        # Backward pass
        if use_amp:
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss.backward()
            optimizer.step()


        # update total epoch loss, iters, l2 dist. with batch outputs
        running_loss += batch_loss.item() # update epoch running loss with batch loss
        running_mse_loss += batch_mse_loss.item()  # Track MSE
        running_residual_loss += batch_residual_loss.item() if use_residual_loss else 0.0
        total_epoch_iters += batch_total_iters
        total_epoch_cg_iters += batch_total_cg_iters
        total_epoch_l2_dist += batch_total_l2_dist
        total_epoch_conv_iters += batch_total_conv_iters
        batch_count += 1
        
        # TensorBoard logging for each step (batch-outputs)
        tb_x = epoch_ix * len(training_loader) + i + 1
        tb_writer.add_scalar('Loss/train', batch_loss.item(), tb_x)
        tb_writer.add_scalar('Loss/train_mse', batch_mse_loss.item(), tb_x)
        if use_residual_loss:
            tb_writer.add_scalar('Loss/train_residual', batch_residual_loss.item(), tb_x)
        tb_writer.add_scalar('Iters/batch', avg_batch_iters, tb_x)
        tb_writer.add_scalar('CG_iters/batch_avg_per_newton', avg_batch_cg_per_newton, tb_x)
        tb_writer.add_scalar('L2_dist/batch', avg_batch_l2_dist, tb_x)
        if track_convergence:
            tb_writer.add_scalar('Conv_iters/batch', avg_batch_conv_iters, tb_x)

    # Return epoch statistics
    avg_epoch_loss = running_loss / batch_count if batch_count > 0 else 0.
    avg_epoch_mse_loss = running_mse_loss / batch_count if batch_count > 0 else 0.
    avg_epoch_residual_loss = running_residual_loss / batch_count if (use_residual_loss and batch_count > 0) else 0.
    total_samples = sum(len(batch[0]) for batch in training_loader)
    avg_epoch_iters = total_epoch_iters / total_samples if total_samples > 0 else 0
    avg_epoch_cg_per_newton = total_epoch_cg_iters / max(total_epoch_iters, 1)
    avg_epoch_l2_dist = total_epoch_l2_dist / total_samples if total_samples > 0 else 0.0
    avg_epoch_conv_iters = total_epoch_conv_iters / total_samples if (track_convergence and total_samples > 0) else 0.0

    # Three-state decision: increase, decrease, or keep maxiters
    maxiters_action = 'keep'  # Default: no change
    
    if avg_epoch_l2_dist < delta_M_low:
        # Very low error: model predictions are excellent, can increase maxiters
        maxiters_action = 'increase'
        print(f'Flag to INCREASE maxiters in next epoch (L2={avg_epoch_l2_dist:.6e} < {delta_M_low:.6e})')
    elif avg_epoch_l2_dist > delta_M_high:
        # High error: solver needs more iterations, decrease maxiters to allow more refinement
        maxiters_action = 'decrease'
        print(f'Flag to DECREASE maxiters in next epoch (L2={avg_epoch_l2_dist:.6e} > {delta_M_high:.6e})')
    else:
        # In optimal range: keep current maxiters
        maxiters_action = 'keep'
        print(f'Maxiters in optimal range (L2={avg_epoch_l2_dist:.6e} in [{delta_M_low:.6e}, {delta_M_high:.6e}])')
    
    return avg_epoch_loss, avg_epoch_iters, avg_epoch_cg_per_newton, avg_epoch_l2_dist, maxiters_action, avg_epoch_mse_loss, avg_epoch_conv_iters, avg_epoch_residual_loss


def val_epoch(model, epoch_ix, loss_fn, val_loader, tb_writer, precision_config=None):
    running_loss = 0.
    running_mse_loss = 0.
    batch_count = 0
    
    device = next(model.parameters()).device
    _dtype = precision_config['input_dtype'] if precision_config else torch.float32

    # Create MSE loss function for logging
    mse_loss_fn = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        for i, (input_tensor, solution_tensor, sample_idx) in enumerate(val_loader):
            input_tensor = input_tensor.to(device=device, dtype=_dtype)
            solution_tensor = solution_tensor.squeeze(1).to(device=device, dtype=_dtype)
            
            # Forward pass
            pred = model(input_tensor)
            pred = pred.squeeze(1)
            
            # Compute loss
            #print(f'IN val_epoch')
            #print(pred.shape, solution_tensor.shape)
            # Verify shapes match
            assert pred.shape == solution_tensor.shape, \
                f"Shape mismatch: pred {pred.shape} vs target {solution_tensor.shape}"
            
            batch_loss = loss_fn(pred, solution_tensor)
            batch_mse_loss = mse_loss_fn(pred, solution_tensor)  # MSE for logging
            
            running_loss += batch_loss.item()
            running_mse_loss += batch_mse_loss.item()  # Track MSE
            batch_count += 1
            
            # FIX: Use separate namespace and correct global step calculation
            # Global step should be: total_batches_processed across all epochs
            total_train_batches_per_epoch = len(tb_writer.train_loader) if hasattr(tb_writer, 'train_loader') else 1
            tb_x = epoch_ix * total_train_batches_per_epoch + (epoch_ix * len(val_loader)) + i + 1
            tb_writer.add_scalar('Loss/val', batch_loss.item(), tb_x)
            tb_writer.add_scalar('Loss/val_mse', batch_mse_loss.item(), tb_x)  # Log MSE
    
    # Return average loss over all batches
    avg_val_loss = running_loss / batch_count if batch_count > 0 else 0.
    avg_val_mse_loss = running_mse_loss / batch_count if batch_count > 0 else 0.
    return avg_val_loss, avg_val_mse_loss


def val_epoch_solver(model, epoch_ix, loss_fn, val_loader, tb_writer, x_pts, y_pts, maxiters, solver_pool, num_processes=2, precision_config=None, solver_config=None, residual_config=None):
    """
    Solver-based validation: mirrors training loop without backward pass.
    Runs model prediction -> solver refinement -> loss(pred, solver_output).
    Optionally adds FEM residual loss when residual_config.use_residual_loss is True.
    No ground truth needed — validation target is the solver output, same as training.

    Returns:
        avg_val_loss, avg_val_mse_loss, avg_val_l2_dist, avg_val_iters, avg_val_conv_iters, avg_val_residual_loss
    """
    running_loss = 0.
    running_mse_loss = 0.
    running_residual_loss = 0.
    total_l2_dist = 0.0
    total_iters = 0
    total_conv_iters = 0
    total_samples = 0
    batch_count = 0

    device = next(model.parameters()).device
    _dtype = precision_config['input_dtype'] if precision_config else torch.float32
    _sol_dtype = precision_config['solver_output_torch_dtype'] if precision_config else torch.float32
    mse_loss_fn = nn.MSELoss()

    # Residual loss config
    if residual_config is None:
        residual_config = {}
    use_residual_loss = bool(residual_config.get('use_residual_loss', False))
    lambda_residual   = float(residual_config.get('lambda_residual', 1.0))
    if use_residual_loss:
        from utils_train_residual import FEMResidualLoss  # lazy to avoid circular import
    residual_loss_fn  = FEMResidualLoss() if use_residual_loss else None

    # CG solver config
    if solver_config is None:
        solver_config = {}
    _linear_solver = solver_config.get('linear_solver', 'direct')
    _cg_maxiter = int(solver_config.get('cg_maxiter', 1000))
    _cg_tol = float(solver_config.get('cg_tol', 1e-12))
    _track_convergence = bool(solver_config.get('track_convergence', False))
    _convergence_maxiters = int(solver_config.get('convergence_maxiters', maxiters))
    _newton_dampfactor = float(solver_config.get('newton_dampfactor', 1.0))

    model.eval()
    with torch.no_grad():
        for i, (input_tensor, alpha, f, mu1, mu2, n, bcs, bc_values, sample_idxs) in enumerate(val_loader):
            input_tensor = input_tensor.to(device=device, dtype=_dtype)
            pred = model(input_tensor)
            pred_numpy = pred.squeeze(1).detach().to(dtype=_sol_dtype).cpu().numpy()
            batch_size = input_tensor.shape[0]
            _n_pts = pred_numpy.shape[-1]

            # Prepare solver args (same as training)
            solver_args = []
            for batch_idx in range(batch_size):
                _bcs_1d = bcs[batch_idx].tolist() if isinstance(bcs[batch_idx], torch.Tensor) else bcs[batch_idx]
                _bc_val = float(bc_values[batch_idx])
                _pred_init = apply_bc_mask_np(pred_numpy[batch_idx], _bcs_1d, _bc_val, _n_pts)
                solver_args.append({
                    'u_init': _pred_init,
                    'k': float(alpha[batch_idx]),
                    'n': int(n[batch_idx]),
                    'A': float(f[batch_idx]),
                    'mu1': float(mu1[batch_idx]),
                    'mu2': float(mu2[batch_idx]),
                    'dirichlet_bcs': _bcs_1d,
                    'dirichlet_value': _bc_val,
                    'x_pts': x_pts,
                    'y_pts': y_pts,
                    'maxiters': maxiters,
                    'verbose': False,
                    'sample_idx': int(sample_idxs[batch_idx]),
                    'batch_idx': batch_idx,
                    'linear_solver': _linear_solver,
                    'cg_maxiter': _cg_maxiter,
                    'cg_tol': _cg_tol,
                    'track_convergence': _track_convergence,
                    'convergence_maxiters': _convergence_maxiters,
                    'dampfactor': _newton_dampfactor,
                })

            # Submit solver tasks to pool
            futures = []
            for args_dict in solver_args:
                future = solver_pool.submit(process_single_sample, args_dict)
                futures.append((future, args_dict['sample_idx'], args_dict['batch_idx']))

            # Collect results
            u_sols = [None] * batch_size
            batch_total_iters = 0
            batch_total_l2_dist = 0.0
            batch_total_conv_iters = 0

            for future, sample_idx, batch_idx in futures:
                try:
                    result = future.result(timeout=60)
                    if result['status'] == 'success':
                        u_sols[batch_idx] = result['u_sol']
                        batch_total_iters += result['iters']
                        batch_total_l2_dist += result['l2_dist']
                        if result.get('conv_iters') is not None:
                            batch_total_conv_iters += result['conv_iters']
                    else:
                        print(f"[Val] Solver failed for sample {sample_idx}: {result.get('error', 'Unknown')}")
                        u_sols[batch_idx] = pred_numpy[batch_idx]
                        batch_total_iters += maxiters
                except concurrent.futures.TimeoutError:
                    print(f"[Val] Solver timeout for sample {sample_idx}")
                    u_sols[batch_idx] = pred_numpy[batch_idx]
                    batch_total_iters += maxiters
                except Exception as e:
                    print(f"[Val] Solver error for sample {sample_idx}: {str(e)}")
                    u_sols[batch_idx] = pred_numpy[batch_idx]
                    batch_total_iters += maxiters

            # Compute loss(pred, solver_output)
            valid_samples = []
            valid_preds   = []
            valid_bcs     = []
            valid_bc_vals = []
            valid_alpha   = []
            valid_f       = []
            valid_mu1     = []
            valid_mu2     = []
            valid_n       = []
            for batch_idx in range(batch_size):
                if u_sols[batch_idx] is not None:
                    u_sol_tensor = torch.tensor(u_sols[batch_idx], dtype=_sol_dtype, device=device)
                    if u_sol_tensor.dim() == 2:
                        u_sol_tensor = u_sol_tensor.unsqueeze(0)
                    valid_samples.append(u_sol_tensor)
                    valid_preds.append(pred[batch_idx:batch_idx+1].squeeze(1))
                    valid_bcs.append(bcs[batch_idx].unsqueeze(0) if isinstance(bcs[batch_idx], torch.Tensor) else torch.tensor([bcs[batch_idx]]))
                    valid_bc_vals.append(float(bc_values[batch_idx]))
                    valid_alpha.append(float(alpha[batch_idx]))
                    valid_f.append(float(f[batch_idx]))
                    valid_mu1.append(float(mu1[batch_idx]))
                    valid_mu2.append(float(mu2[batch_idx]))
                    valid_n.append(float(n[batch_idx]))

            batch_residual_loss = torch.tensor(0.0, device=device)
            if len(valid_samples) > 0:
                u_sol_batch  = torch.cat(valid_samples,  dim=0)
                pred_batch   = torch.cat(valid_preds,    dim=0)
                _vbcs_t      = torch.cat(valid_bcs,      dim=0).to(device=device)
                _vbc_vals_t  = torch.tensor(valid_bc_vals, device=device, dtype=pred_batch.dtype)
                pred_batch   = apply_bc_mask_torch_batch(pred_batch, _vbcs_t, _vbc_vals_t, pred_batch.shape[-1])
                batch_loss      = loss_fn(pred_batch, u_sol_batch)
                batch_mse_loss  = mse_loss_fn(pred_batch, u_sol_batch)

                if use_residual_loss:
                    _k_t   = torch.tensor(valid_alpha, device=device, dtype=pred_batch.dtype)
                    _A_t   = torch.tensor(valid_f,     device=device, dtype=pred_batch.dtype)
                    _mu1_t = torch.tensor(valid_mu1,   device=device, dtype=pred_batch.dtype)
                    _mu2_t = torch.tensor(valid_mu2,   device=device, dtype=pred_batch.dtype)
                    _n_t   = torch.tensor(valid_n,     device=device, dtype=pred_batch.dtype)
                    batch_residual_loss = residual_loss_fn(
                        pred_batch, _k_t, _n_t, _A_t, _mu1_t, _mu2_t, x_pts, y_pts)
                    batch_loss = batch_loss + lambda_residual * batch_residual_loss
            else:
                batch_loss = torch.tensor(0.0, device=device)
                batch_mse_loss = torch.tensor(0.0, device=device)

            running_loss += batch_loss.item()
            running_mse_loss += batch_mse_loss.item()
            running_residual_loss += batch_residual_loss.item()
            total_l2_dist += batch_total_l2_dist
            total_iters += batch_total_iters
            total_conv_iters += batch_total_conv_iters
            total_samples += batch_size
            batch_count += 1

            # TensorBoard logging
            total_train_batches_per_epoch = len(tb_writer.train_loader) if hasattr(tb_writer, 'train_loader') else 1
            tb_x = epoch_ix * total_train_batches_per_epoch + (epoch_ix * len(val_loader)) + i + 1
            tb_writer.add_scalar('Loss/val', batch_loss.item(), tb_x)
            tb_writer.add_scalar('Loss/val_mse', batch_mse_loss.item(), tb_x)
            tb_writer.add_scalar('L2_dist/val_batch', batch_total_l2_dist / batch_size, tb_x)
            if use_residual_loss:
                tb_writer.add_scalar('Loss/val_residual', batch_residual_loss.item(), tb_x)
            if _track_convergence:
                tb_writer.add_scalar('Conv_iters/val_batch', batch_total_conv_iters / batch_size, tb_x)

    avg_val_loss = running_loss / batch_count if batch_count > 0 else 0.
    avg_val_mse_loss = running_mse_loss / batch_count if batch_count > 0 else 0.
    avg_val_residual_loss = running_residual_loss / batch_count if (use_residual_loss and batch_count > 0) else 0.
    avg_val_l2_dist = total_l2_dist / total_samples if total_samples > 0 else 0.
    avg_val_iters = total_iters / total_samples if total_samples > 0 else 0.
    avg_val_conv_iters = total_conv_iters / total_samples if (_track_convergence and total_samples > 0) else 0.0
    return avg_val_loss, avg_val_mse_loss, avg_val_l2_dist, avg_val_iters, avg_val_conv_iters, avg_val_residual_loss


def train_model_mp(model, training_loader, validation_loader, x_pts, y_pts, writer=None, config = None):
    losses = []
    val_losses = []
    nr_iters = []
    nr_conv_iters = []
    epoch_number = 0
    best_loss = float('inf')
    epoch_times = []
    total_train_time = 0.0

    if config is None:
        config = {}
    config = resolve_run_paths(config)
    training_config = config['training_config']
    precision_config = get_precision_config(config)

    # Get initial maxiters and delta_M threshold
    maxiters = int(config['solver_config'].get('maxiters'))
    delta_M_high = float(config['solver_config'].get('delta_M_high'))
    delta_M_low = float(config['solver_config'].get('delta_M_low'))
    max_maxiters = int(config['solver_config'].get('max_maxiters'))
    min_maxiters = int(config['solver_config'].get('min_maxiters'))
    solver_iteration_mode = config['solver_config'].get('solver_iteration_mode', 'fixed')

    # Patience parameters to avoid oscillations
    increase_patience = int(config['solver_config'].get('increase_patience'))  # Wait N epochs before increasing
    decrease_patience = int(config['solver_config'].get('decrease_patience'))  # Wait N epochs before decreasing
    
    increase_counter = 0
    decrease_counter = 0

    print(f"Solver iteration mode: {solver_iteration_mode}")
    print(f"Adaptive maxiters configuration:")
    print(f"  Starting maxiters: {maxiters}")
    print(f"  Range: [{min_maxiters}, {max_maxiters}]")
    print(f"  L2 thresholds: low={delta_M_low}, high={delta_M_high}")
    print(f"  Patience: increase={increase_patience}, decrease={decrease_patience}")

    # Get inference configuration
    inference_interval = config.get('training_config', {}).get('inference_interval')
    test_sample_indices = config.get('training_config', {}).get('test_sample_indices')
    # Get test dataset
    inference_path = config.get('data_config', {}).get('train_path', {})
    inference_dataset = PDEDataset(inference_path, config, sample_type='train')
    
    print(f"Periodic inference enabled: every {inference_interval} epochs on samples {test_sample_indices}")

    # Initialize solver pool
    num_solver_processes = config['training_config'].get('num_solver_processes')  # Adjust based on CPU cores
    solver_pool = concurrent.futures.ProcessPoolExecutor(max_workers=num_solver_processes)
    print(f"Created solver pool with {num_solver_processes} workers")
     

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = training_config.get('checkpoint_dir')
    run_dir = training_config.get('run_dir', os.path.dirname(checkpoint_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    # Save config to run_dir root
    if config is not None:
        config_save_path = os.path.join(run_dir, 'config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Config saved to {config_save_path}")

    # --- MLflow ---
    mlflow_run = start_mlflow_run(config)
    log_mlflow_config(mlflow_run, config, run_dir)

    # --- CSV metrics logger ---
    csv_path = os.path.join(run_dir, 'metrics.csv')
    _csv_fields = ['epoch', 'train_nmse', 'val_nmse', 'train_mse', 'val_mse',
                   'newton_iters', 'cg_per_newton', 'maxiters', 'epoch_time_s', 'train_residual_nmse', 'val_residual_nmse']
    _csv_file   = open(csv_path, 'w', newline='')
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields)
    _csv_writer.writeheader()
    print(f"CSV metrics logger: {csv_path}")
    
    # write configs to tensorboard
    # Log config to TensorBoard
    if writer is not None and config is not None:
        # Convert config to formatted string
        config_text = yaml.dump(config, default_flow_style=False, sort_keys=False)
        writer.add_text('Config/YAML', f'```yaml\n{config_text}\n```', 0)
        
        # Also log as individual hyperparameters
        writer.add_hparams(
            hparam_dict={
                'lr': training_config.get('learning_rate'),
                'batch_size': training_config.get('batch_size'),
                'epochs': training_config.get('epochs'),
                'precision': precision_config['precision'],
                'maxiters': config['solver_config'].get('maxiters'),
                'n_pts': config['data_config'].get('n_pts'),
                'train_samples': int(config['data_config'].get('total_num_samples')*config['data_config'].get('train_frac')),
                'val_samples': int(config['data_config'].get('total_num_samples')*config['data_config'].get('val_frac')),
                'in_channels': config['model_config'].get('in_channels'),
                'out_channels': config['model_config'].get('out_channels'),
                'scheduler': training_config['use_lr_scheduler'],
                'scheduler_step': training_config['scheduler_config'].get('step_size'),
                'scheduler_gamma': training_config['scheduler_config'].get('gamma'),
            },
            metric_dict={
                'hparam/placeholder': 0
            }
        )
        print("Config logged to TensorBoard")

    # loss function and optimizer (fixed lr)
    
    loss_fn = NormalizedMSELoss() #torch.nn.MSELoss()
    lr = training_config.get('learning_rate')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=training_config.get('weight_decay'))

    # Initialize gradient scaler for mixed precision
    device = next(model.parameters()).device
    use_amp = precision_config['use_amp']
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f"Training precision: {precision_config['precision']} | Model dtype: {precision_config['model_dtype']} | AMP: {'on' if use_amp else 'off'} | Solver: fp64 (always)")
    
    use_lr_scheduler = training_config.get('use_lr_scheduler')
    if use_lr_scheduler == True:
        step_size = training_config.get('scheduler_config').get('step_size')
        gamma = training_config.get('scheduler_config').get('gamma')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Load checkpoint if exists
    load_checkpoint_path = training_config.get('load_checkpoint_path')

    if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path):
        checkpoint = torch.load(load_checkpoint_path, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        state_dict.pop('_metadata', None)  # strip neuraloperator-format key if present
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if use_lr_scheduler == True:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_number = checkpoint['epoch']
        losses = checkpoint['losses']
        val_losses = checkpoint.get('val_losses', [])
        nr_iters = checkpoint['nr_iters']
        nr_conv_iters = checkpoint.get('nr_conv_iters', [])
        epoch_times = checkpoint.get('epoch_times', [])
        total_train_time = sum(epoch_times)
        best_loss = checkpoint['best_loss']

    if writer is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/trainer_{}'.format(timestamp))

    # Store training_loader reference for validation logging
    writer.train_loader = training_loader

    epochs = training_config.get('epochs')
    tol = config['solver_config'].get('tol')
    solver_config = config.get('solver_config', {})

    _ls = solver_config.get('linear_solver', 'direct')
    print(f"Newton linear solver: {_ls}" + (f" (CG maxiter={solver_config.get('cg_maxiter', 1000)}, tol={solver_config.get('cg_tol', 1e-12)})" if _ls == 'cg' else ''))
    
    # MAIN TRAINING LOOP
    for epoch in range(epoch_number, epochs):
        print('EPOCH {}:'.format(epoch + 1))
        print(f'Current maxiters: {maxiters}')
        
        t0 = time.time()
        model.train(True)
        avg_loss, n_iters, avg_cg_per_newton, avg_l2_dist, maxiters_action, avg_mse_loss, avg_conv_iters, avg_residual_loss = train_epoch_mp(model, epoch, optimizer, loss_fn, training_loader,
                                                       writer, x_pts, y_pts, maxiters, scaler=scaler, solver_pool=solver_pool, num_processes=num_solver_processes,
                                                       delta_M_high=delta_M_high, delta_M_low=delta_M_low, precision_config=precision_config,
                                                       solver_config=solver_config, residual_config=config.get('residual_config', {}))
        
        losses.append(avg_loss)
        nr_iters.append(n_iters)
        if solver_config.get('track_convergence'):
            nr_conv_iters.append(avg_conv_iters)

        # Solver-based validation (no ground truth needed — mirrors training)
        model.eval()
        avg_val_loss, avg_val_mse_loss, avg_val_l2_dist, avg_val_iters, avg_val_conv_iters, avg_val_residual_loss = val_epoch_solver(
            model, epoch, loss_fn, validation_loader, writer,
            x_pts, y_pts, maxiters, solver_pool, num_processes=num_solver_processes, precision_config=precision_config,
            solver_config=solver_config, residual_config=config.get('residual_config', {}))
        val_losses.append(avg_val_loss)
        
        # Print epoch summary
        print(f'LOSS train: {avg_loss:.6f}, val: {avg_val_loss:.6f}, '
              f'iters: {n_iters:.1f}, L2_dist: {avg_l2_dist:.6e}')
        print(f'MSE  train: {avg_mse_loss:.6f}, val: {avg_val_mse_loss:.6f}')
        print(f'Val  L2_dist: {avg_val_l2_dist:.6e}, val_iters: {avg_val_iters:.1f}')
        if solver_config.get('track_convergence'):
            print(f'Conv iters (to convergence): train={avg_conv_iters:.1f}, val={avg_val_conv_iters:.1f}')
        print(f'CG   avg_iters/newton_step: {avg_cg_per_newton:.2f}')
        
        epoch_dt = time.time() - t0
        total_train_time += epoch_dt
        epoch_times.append(epoch_dt)
        print(f'  Time: {epoch_dt:.2f}s  (total: {total_train_time:.1f}s)')

        # *** ADAPTIVE MAXITERS WITH PATIENCE ***
        old_maxiters = maxiters

        if solver_iteration_mode == 'adaptive':
            if maxiters_action == 'increase':
                increase_counter += 1
                decrease_counter = 0

                if increase_counter >= increase_patience and maxiters < max_maxiters:
                    maxiters = min(maxiters * 2, max_maxiters)
                    print(f'\n>>> INCREASING maxiters: {old_maxiters} -> {maxiters}')
                    print(f'    Reason: L2 distance ({avg_l2_dist:.6e}) < low threshold ({delta_M_low:.6e}) for {increase_counter} epochs')
                    increase_counter = 0
                    writer.add_scalar('Training/maxiters_change', 1, epoch + 1)

            elif maxiters_action == 'decrease':
                decrease_counter += 1
                increase_counter = 0

                if decrease_counter >= decrease_patience and maxiters > min_maxiters:
                    maxiters = max(maxiters // 2, min_maxiters)
                    print(f'\n>>> DECREASING maxiters: {old_maxiters} -> {maxiters}')
                    print(f'    Reason: L2 distance ({avg_l2_dist:.6e}) > high threshold ({delta_M_high:.6e}) for {decrease_counter} epochs')
                    decrease_counter = 0
                    writer.add_scalar('Training/maxiters_change', -1, epoch + 1)

            else:  # 'keep'
                increase_counter = 0
                decrease_counter = 0
                print(f'>>> Keeping maxiters={maxiters} (L2 in optimal range)')

            print(f'    Counters: increase={increase_counter}/{increase_patience}, decrease={decrease_counter}/{decrease_patience}\n')
        else:  # 'fixed'
            print(f'>>> Fixed maxiters={maxiters} (solver_iteration_mode=fixed)\n')

        # Write epoch-level metrics to tensorboard
        writer.add_scalars('Loss/epoch', {
            'train': avg_loss,
            'val': avg_val_loss
        }, epoch + 1)
        writer.add_scalars('Loss_MSE/epoch', {
            'train': avg_mse_loss,
            'val': avg_val_mse_loss
        }, epoch + 1)
        if config.get('residual_config', {}).get('use_residual_loss'):
            writer.add_scalar('Loss/train_residual_epoch', avg_residual_loss, epoch + 1)
            writer.add_scalar('Loss/val_residual_epoch', avg_val_residual_loss, epoch + 1)
        writer.add_scalar('Iters/epoch', n_iters, epoch + 1)
        writer.add_scalar('CG_iters/epoch_avg_per_newton', avg_cg_per_newton, epoch + 1)
        if solver_config.get('track_convergence'):
            writer.add_scalars('Conv_iters/epoch', {'train': avg_conv_iters, 'val': avg_val_conv_iters}, epoch + 1)
        writer.add_scalars('L2_dist/epoch', {
            'train': avg_l2_dist,
            'val': avg_val_l2_dist
        }, epoch + 1)
        writer.add_scalar('Training/maxiters', maxiters, epoch + 1)
        writer.add_scalar('Training/increase_counter', increase_counter, epoch + 1)
        writer.add_scalar('Training/decrease_counter', decrease_counter, epoch + 1)
        writer.add_scalar('Time/epoch_seconds', epoch_dt, epoch + 1)
        writer.add_scalar('Time/total_seconds', total_train_time, epoch + 1)
        writer.flush()

        log_mlflow_metrics(mlflow_run, {
            'train_nmse':          avg_loss,
            'val_nmse':            avg_val_loss,
            'train_mse':           avg_mse_loss,
            'val_mse':             avg_val_mse_loss,
            'train_l2_dist':       avg_l2_dist,
            'val_l2_dist':         avg_val_l2_dist,
            'newton_iters':        n_iters,
            'cg_per_newton':       avg_cg_per_newton,
            'maxiters':            float(maxiters),
            'epoch_time':          epoch_dt,
            **({'train_residual_nmse': avg_residual_loss, 'val_residual_nmse': avg_val_residual_loss} if config.get('residual_config', {}).get('use_residual_loss') else {}),
            **({'train_conv_iters': avg_conv_iters, 'val_conv_iters': avg_val_conv_iters} if solver_config.get('track_convergence') else {}),
        }, step=epoch + 1)        

        # --- CSV ---
        _csv_writer.writerow({
            'epoch':                epoch + 1,
            'train_nmse':           avg_loss,
            'val_nmse':             avg_val_loss,
            'train_mse':            avg_mse_loss,
            'val_mse':              avg_val_mse_loss,
            'newton_iters':         n_iters,
            'cg_per_newton':        avg_cg_per_newton,
            'maxiters':             float(maxiters),
            'epoch_time_s':         epoch_dt,
            'train_residual_nmse':  avg_residual_loss if config.get('residual_config', {}).get('use_residual_loss') else '',
            'val_residual_nmse':    avg_val_residual_loss if config.get('residual_config', {}).get('use_residual_loss') else '',
        })
        _csv_file.flush()
        
        if use_lr_scheduler:
            scheduler.step()

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_{epoch + 1}.pt')
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'val_losses': val_losses,
            'nr_iters': nr_iters,
            'nr_conv_iters': nr_conv_iters,
            'epoch_times': epoch_times,
            'best_loss': best_loss
        }
        if use_lr_scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save checkpoint at multiples of 10
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, checkpoint_path)
            print(f'  Checkpoint saved at epoch {epoch + 1}')
        
        # Keep track of top 3 checkpoints based on validation loss
        if not hasattr(train_model_mp, 'top_checkpoints'):
            train_model_mp.top_checkpoints = []  # List of (val_loss, epoch, path) tuples
        
        # Add current checkpoint to candidates
        train_model_mp.top_checkpoints.append((avg_val_loss, epoch + 1, checkpoint_path))
        
        # Sort by validation loss (ascending) and keep top 3
        train_model_mp.top_checkpoints.sort(key=lambda x: x[0])
        
        # Save top 3 checkpoints
        if len(train_model_mp.top_checkpoints) > 3:
            removed = train_model_mp.top_checkpoints.pop()
            removed_path = removed[2]
            removed_epoch = removed[1]
            if os.path.exists(removed_path) and removed_epoch % 10 != 0:
                os.remove(removed_path)
                print(f'  Removed checkpoint from epoch {removed_epoch}')
        
        # Save if in top 3
        if any(epoch + 1 == ckpt[1] for ckpt in train_model_mp.top_checkpoints):
            torch.save(checkpoint, checkpoint_path)
            print(f'  Top-3 checkpoint saved at epoch {epoch + 1} (val_loss: {avg_val_loss:.6f})')

        # Save best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            print(f'  New best model saved with val loss: {best_loss:.6f}')
            log_mlflow_artifact(mlflow_run, best_model_path)

        # PERIODIC INFERENCE ON TEST SAMPLES
        if (epoch + 1) % inference_interval == 0:
            print(f"\n{'='*60}")
            print(f"Running inference at epoch {epoch + 1}")
            print(f"{'='*60}")
            
            # Create inference directory
            inference_plot_dir = os.path.join(run_dir, 'inference_plots')
            os.makedirs(inference_plot_dir, exist_ok=True)
        
            inference_metrics = periodic_inference(
                model=model,
                test_dataset=inference_dataset,
                sample_indices=test_sample_indices,
                x_pts=x_pts,
                y_pts=y_pts,
                config=config,
                epoch=epoch + 1,
                plot_dir=inference_plot_dir,
                device=device,
                writer=writer
            )
            log_mlflow_artifact(mlflow_run, os.path.join(inference_plot_dir, f'epoch_{epoch + 1}'))

            print(f"{'='*60}\n")        

    # Cleanup pool
    print("Shutting down solver pool...")
    solver_pool.shutdown(wait=True)
    print("Solver pool shut down successfully")    


    print(f"\nTraining complete. Total time: {total_train_time:.1f}s "
          f"({total_train_time / max(epochs, 1):.2f}s/epoch avg)")

    _csv_file.close()
    print(f"Metrics CSV closed: {csv_path}")
    log_mlflow_artifact(mlflow_run, csv_path)

    end_mlflow_run(mlflow_run)
    return losses, val_losses, nr_iters, nr_conv_iters, epoch_times, model






