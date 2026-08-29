import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import yaml

import scienceplots
plt.style.use(['science', 'grid', 'no-latex'])
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12

from model_pdes import *
from utils_plots import *
from utils_train_residual import (
    parse_args,
    resolve_run_paths,
    get_precision_config,
    create_model,
    convert_model_precision,
    PDEDataset,
    train_model_residual,
)
from solver import sampling_points_2d_grid


# ============================================================================
# Main
# ============================================================================

def main():
    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')

    # --- Config ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = resolve_run_paths(config)

    print("\n=== YAML Configuration ===")
    with open(args.config, 'r') as f:
        print(f.read())
    print("==========================\n")

    n_pts = config['data_config']['n_pts']
    x_pts, y_pts, xv, yv, nx, ny = sampling_points_2d_grid(n_pts=n_pts)

    # --- Datasets ---
    # training_loader: sample_type='train' — yields PDE params for residual loss
    # validation_loader: sample_type='test' — yields ground-truth solution for supervised val
    train_path = config['data_config']['train_path']
    val_path   = config['data_config']['val_path']

    print(f"Data paths:")
    print(f"  Train (residual): {train_path}")
    print(f"  Val  (supervised): {val_path}")

    train_dataset = PDEDataset(train_path, config, sample_type='train')
    val_dataset   = PDEDataset(val_path,   config, sample_type='test')

    batch_size   = config['training_config']['batch_size']
    num_workers  = config['training_config'].get('num_data_workers', 4)
    pin          = device.type == 'cuda'

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    # --- TensorBoard ---
    timestamp      = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_path = config['training_config'].get('log_path')
    log_name = config['training_config'].get('log_name', 'residual')
    if log_path:
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(os.path.join(log_path, f'{log_name}_{timestamp}'))
    else:
        writer = SummaryWriter(f'runs_pdes/residual_{timestamp}')

    # --- Model ---
    precision_config = get_precision_config(config)
    model = create_model(config)
    model = convert_model_precision(model, precision_config['model_dtype'])
    model = model.to(device=device)

    print(f"\nTraining precision: {precision_config['precision']} | "
          f"Model dtype: {precision_config['model_dtype']} | Mode: residual")
    print("\n=== Model ===")
    print(model)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 50 + "\n")

    losses, val_nmses, epoch_times, model = train_model_residual(
        model, train_loader, val_loader,
        x_pts=x_pts, y_pts=y_pts,
        writer=writer, config=config,
    )

    # --- Post-training plots ---
    plot_dir = f'./plots_pdes/residual_{plot_timestamp}'
    os.makedirs(plot_dir, exist_ok=True)
    plot_loss_curve(losses, len(losses), plot_dir, val_losses=val_nmses)

    plt.figure()
    plt.plot(range(1, len(epoch_times) + 1), epoch_times, color='black')
    plt.xlabel('epoch')
    plt.ylabel('wall-clock time (s)')
    plt.grid()
    plt.savefig(f'{plot_dir}/epoch_time.pdf', format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
