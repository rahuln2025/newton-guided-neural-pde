
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
from utils_train_datadriven import *

import h5py
import yaml
import argparse
from torchinfo import summary




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
    config = resolve_run_paths(config)  # expand run_name → checkpoint_dir, log_path, log_name

    print("\n=== YAML Configuration ===")
    with open(args.config, 'r') as f:
        print(f.read())
    print("==========================\n")

    n_pts = config['data_config']['n_pts']
    x_pts, y_pts, xv, yv, nx, ny = sampling_points_2d_grid(n_pts=n_pts)

    # --- Datasets ---
    # Supervised: all datasets loaded with sample_type='test' to get solution_tensor
    train_path = config['data_config'].get('supervised_train_path',
                                            config['data_config']['train_path'])
    val_path = config['data_config']['val_path']
    test_path = config['data_config']['test_path']

    print(f"Data paths:")
    print(f"  Train (supervised): {train_path}")
    print(f"  Val:                {val_path}")
    print(f"  Test:               {test_path}")

    train_dataset = PDEDataset(train_path, config, sample_type='test')  # returns (input, solution, idx)
    val_dataset = PDEDataset(val_path, config, sample_type='test')
    test_dataset = PDEDataset(test_path, config, sample_type='train')   # for inference_and_plot (needs PDE params)

    batch_size = config['training_config']['batch_size']
    num_workers = config['training_config'].get('num_data_workers', 4)
    pin = device.type == 'cuda'

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    # --- TensorBoard ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_path = config['training_config'].get('log_path')
    log_name = config['training_config'].get('log_name', 'supervised')
    if log_path:
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(os.path.join(log_path, f'{log_name}_{timestamp}'))
    else:
        writer = SummaryWriter(f'runs/supervised_{timestamp}')

    # --- Model ---
    precision_config = get_precision_config(config)
    model = create_model(config)
    model = convert_model_precision(model, precision_config['model_dtype'])
    model = model.to(device=device)

    print(f"\nTraining precision: {precision_config['precision']} | "
          f"Model dtype: {precision_config['model_dtype']} | Mode: supervised")
    print("\n=== Model ===")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 50 + "\n")

    losses, val_losses, epoch_times, model = train_model_supervised(
                model, train_loader, val_loader,
                writer=writer, config=config,
            )

    # --- Post-training plots ---
    plot_dir = f'./plots_pdes/supervised_{plot_timestamp}'
    os.makedirs(plot_dir, exist_ok=True)
    plot_loss_curve(losses, len(losses), plot_dir, val_losses=val_losses)

    # Epoch time plot
    matplotlib.use('Agg')
    plt.figure()
    plt.plot(range(1, len(epoch_times) + 1), epoch_times, color='black')
    plt.xlabel('epoch')
    plt.ylabel('wall-clock time (s)')
    plt.grid()
    plt.savefig(f'{plot_dir}/epoch_time.pdf', format='pdf')
    plt.close()

if __name__ == "__main__":
    main()


