
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
from utils_train_with_solver import *

import h5py
import yaml
import argparse
from torchinfo import summary



def main():

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(1)}")
        print(f"Memory Available: {torch.cuda.get_device_properties(1).total_memory / 1e9:.2f} GB")

    args = parse_args()

    # Set specific GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    config_path = args.config
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Print all contents of the YAML file
    print("\n=== YAML Configuration File Contents ===")
    with open(config_path, 'r') as file:
        yaml_contents = file.read()
        print(yaml_contents)
    print("======================================\n")

    n_pts = config.get('data_config').get('n_pts')
    x_pts, y_pts, xv, yv, nx, ny = sampling_points_2d_grid(n_pts = n_pts)

    train_path = config.get('data_config').get('train_path')
    val_path = config.get('data_config').get('val_path')
    test_path = config.get('data_config').get('test_path')

    print(f'Data paths: Train: {train_path}, Val: {val_path}')
        
    train_dataset = PDEDataset(train_path, config, sample_type='train')
    val_dataset = PDEDataset(val_path, config, sample_type='train')  # Use 'train' format to get PDE params for solver-based validation
    test_dataset = PDEDataset(test_path, config, sample_type='train')
    batch_size = config.get('training_config').get('batch_size')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True,
                                               num_workers=config.get('training_config').get('num_data_workers'),  
                                             pin_memory = True if device.type == 'cuda' else False)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=False,
                                             num_workers=config.get('training_config').get('num_data_workers'),  
                                             pin_memory = True if device.type == 'cuda' else False)
    
    # Initialize tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_path = config.get('training_config').get('log_path')
    log_name = config.get('training_config').get('log_name')
    if log_path:
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(os.path.join(log_path, f'{log_name}_{timestamp}'))
    else:
        writer = SummaryWriter('runs_pdes/1024_UNet_fxy_{}'.format(timestamp)) 

    #model = ImagePredictorUNet(config=config)
    precision_config = get_precision_config(config)
    model = create_model(config)
    model = convert_model_precision(model, precision_config['model_dtype'])
    model = model.to(device=device)

    # Print model summary and parameter count
    print(f"\nTraining precision: {precision_config['precision']} | Model dtype: {precision_config['model_dtype']} | Solver: fp64 (always)")
    print("\n=== Model Summary ===")
    # in_channels = config.get('model_config').get('in_channels')
    # dummy_input = torch.zeros(1, in_channels, n_pts, n_pts, device=device, dtype=precision_config['model_dtype'])
    # summary(model, input_data=dummy_input)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 50 + "\n")

    # train model (with multiprocessing in Python)
    losses, val_losses, nr_iters, nr_conv_iters, epoch_times, model = train_model_mp(model,
                                    training_loader=train_loader, 
                                    validation_loader = val_loader, 
                                    x_pts=x_pts,
                                    y_pts=y_pts, 
                                    writer=writer, 
                                    config=config)

    # Call plotting functions with data
    # Create plots directory inside the run folder derived by resolve_run_paths
    run_dir = os.path.join(config.get('run_root', './runs'), config.get('run_name', f'run_{plot_timestamp}'))
    plot_dir = os.path.join(run_dir, f'plots_{plot_timestamp}')
    os.makedirs(plot_dir, exist_ok=True)
    plot_loss_curve(losses, len(losses), plot_dir, val_losses=val_losses)
    plot_iterations_curve(nr_iters, len(nr_iters), plot_dir)

    # Epoch time plot
    matplotlib.use('Agg')
    plt.figure()
    plt.plot(range(1, len(epoch_times) + 1), epoch_times, color='black')
    plt.xlabel('epoch')
    plt.ylabel('wall-clock time (s)')
    plt.grid()
    plt.savefig(f'{plot_dir}/epoch_time.pdf', format='pdf')
    plt.close()
    
    # Plot convergence iterations if tracked
    maxiters = config['solver_config'].get('maxiters')
    matplotlib.use('Agg')
    plt.figure()
    if config['solver_config'].get('track_convergence') and len(nr_conv_iters) > 0:
        plt.plot(range(1, len(nr_conv_iters) + 1), nr_conv_iters,
                color='blue', label='full conv.')
    plt.plot(range(1, len(nr_iters) + 1), nr_iters,
            color='red', label=f'M(={maxiters})')
    plt.xlabel('epoch')
    plt.ylabel('Newton method iterations')
    plt.legend()
    plt.grid()
    plt.savefig(f'{plot_dir}/convergence_iters_comparison.pdf', format='pdf')
    plt.close()
    print(f"Convergence iterations plot saved to {plot_dir}/convergence_iters_comparison.pdf")


if __name__ == "__main__":
    main()
