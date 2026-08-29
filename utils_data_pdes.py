import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os

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

#from model import *
from solver import *
import matplotlib
import argparse
import yaml
import h5py
import csv

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

## alpha --> k
## f --> A

# alpha_train = np.load(".././fixed_point_5d_plots/alpha_train_safe_350.npy") # precomputed safe parameter sets for sample generation

def split_safe_set(safe_set_path, out_dir, total_num_samples=None, train_frac=None, val_frac=None, 
                   seed=0):
    """Split the safe parameter set into disjoint train/val/test partitions.
    Returns paths to the three split files."""
    arr = np.load(safe_set_path)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(arr))
    n = total_num_samples
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val
    

    os.makedirs(out_dir, exist_ok=True)
    paths = {
        'train': os.path.join(out_dir, 'safe_train.npy'),
        'valid': os.path.join(out_dir, 'safe_valid.npy'),
        'test':  os.path.join(out_dir, 'safe_test.npy'),
    }
    np.save(paths['train'], arr[perm[:n_train]])
    np.save(paths['valid'], arr[perm[n_train:n_train + n_val]])
    np.save(paths['test'],  arr[perm[n_train + n_val:n_train + n_val + n_test]])
    
    print(f"Split safe set ({n} rows) → "
          f"train={n_train}, valid={n_val}, test={n - n_train - n_val}")
    return paths, n_train, n_val, n_test


def utils_generate_sample(ix, seed, dataset_type='train', param_set=None, config=None, training_type='solver'):

    A = param_set['A']
    k = param_set['k']
    n = param_set['n']
    mu1 = param_set['mu1']
    mu2 = param_set['mu2']

    # homogenous dirichlet boundary conditions with 0 value
    boundary_conditions = [1, 2, 3, 4] # all sides subjected to dirichlet BC
    dirichlet_value = 0.0

    print(f"[{dataset_type}] Sample {ix}: alpha={k:.2f}, n={n}, f={A:.2f}, "
          f"mu1={mu1:.2f}, mu2={mu2:.2f}, BCs={boundary_conditions}, BC_val={dirichlet_value:.2f}")

    # Generate sampling grid
    data_config = config['data_config']
    grid_size = data_config['n_pts']
    x_pts, y_pts, xv, yv, nx, ny = sampling_points_2d_grid(n_pts=grid_size)

    # Create UNet input
    input_tensor = create_unet_input_fxy(
        k=k,
        n=n,
        A=A,
        mu1=mu1,
        mu2=mu2,
        boundary_conditions=boundary_conditions,
        bc_value=dirichlet_value,
        grid_size=grid_size,
        mu_k=data_config['k_mu'],
        std_k=data_config['k_std'],
        mu_n=data_config['n_mu'],
        std_n=data_config['n_std'],
        mu_f=data_config['fxy_mu'],
        std_f=data_config['fxy_std'],
        config=config
    )

    # Solve PDE
    need_solution = (dataset_type != 'train') or (training_type == 'datadriven')
    if need_solution:
        solver_config = config['solver_config']
        u_sol, iters, gfu, cg_iters, improvement, _ = solve_pde(
            u_init=None,
            k=k,
            n=n,
            A=A,
            mu1=mu1,
            mu2=mu2,
            dirichlet_bcs=boundary_conditions,
            dirichlet_value=dirichlet_value,
            x_pts=x_pts,
            y_pts=y_pts,
            maxiters=solver_config['convergence_maxiters'],
            verbose=False,
            linear_solver=solver_config['linear_solver']
        )
        
        print(f"  Solver converged in {iters} iterations")

    # Store parameters
    params_dict = {
        'k': k,
        'n': n,
        'A': A,
        'mu1': mu1,
        'mu2': mu2,
        'boundary_conditions': boundary_conditions,
        'dirichlet_value': dirichlet_value,
    }

    if need_solution:
        return input_tensor, params_dict, u_sol, iters, cg_iters
    else:
        return input_tensor, params_dict
 


def compute_fxy_stats(param_rows, grid_size, config):
    """
    Compute global standardization stats over the training parameter rows.

    Computes mean and std for:
        - k   (scalar per sample)
        - n   (scalar per sample)
        - f(x,y) = A*(mu1²+mu2²+x²+y²)*sin(mu1πx)*sin(mu2πy)  (field per sample)

    Args:
        param_rows: iterable of dicts with keys k, n, A, mu1, mu2
        grid_size:  spatial resolution
        config:     config dict (unused, kept for interface consistency)

    Returns:
        mu_k, std_k, mu_n, std_n, mu_f, std_f  (all float)
    """
    x = np.linspace(0, 1, grid_size, dtype=np.float64)
    y = np.linspace(0, 1, grid_size, dtype=np.float64)
    x, y = np.meshgrid(x, y)

    k_vals, n_vals = [], []
    f_sum, f_sum2, n_pixels = 0.0, 0.0, 0

    for p in param_rows:
        k_vals.append(float(p['k']))
        n_vals.append(float(p['n']))
        A, mu1, mu2 = float(p['A']), float(p['mu1']), float(p['mu2'])
        f_xy   = A * (mu1**2 + mu2**2 + x**2 + y**2) * np.sin(mu1 * np.pi * x) * np.sin(mu2 * np.pi * y)
        f_sum  += f_xy.sum()
        f_sum2 += (f_xy ** 2).sum()
        n_pixels += f_xy.size

    k_arr = np.array(k_vals)
    n_arr = np.array(n_vals)
    mu_k,  std_k  = float(k_arr.mean()), float(k_arr.std())
    mu_n,  std_n  = float(n_arr.mean()), float(n_arr.std())
    mu_f  = f_sum / n_pixels
    std_f = float(np.sqrt(f_sum2 / n_pixels - mu_f ** 2))
    mu_f  = float(mu_f)
    return mu_k, std_k, mu_n, std_n, mu_f, std_f


def create_unet_input_fxy(k, n, A, mu1, mu2, boundary_conditions, bc_value, grid_size=256,
                          mu_k=None, std_k=None, mu_n=None, std_n=None,
                          mu_f=None, std_f=None, config=None):
    """
    Creates 4-channel input for UNet2D.

    All scalar parameters are globally standardised with training-set statistics.

    Channels:
        0  k_std   — (k - mu_k) / std_k
        1  n_std   — (n - mu_n) / std_n
        2  f_std   — f(x,y) = A*(mu1²+mu2²+x²+y²)*sin(mu1πx)*sin(mu2πy)
                    globally standardised: (f - mu_f) / std_f
        3  bc_mask — Dirichlet boundary mask (0 interior, bc_value on active edges)

    Args:
        k, n, A, mu1, mu2  : PDE parameters (scalars)
        boundary_conditions: list of active boundary indices from {1,2,3,4}
        bc_value           : Dirichlet value applied to active boundaries
        grid_size          : spatial resolution
        mu_k, std_k        : training-set mean and std of k
        mu_n, std_n        : training-set mean and std of n
        mu_f, std_f        : training-set mean and std of f(x,y) pixels
        config             : config dict (kept for interface consistency)

    Returns:
        input_tensor: np.ndarray [4, grid_size, grid_size], float32
    """
    k_std = (k - mu_k) / std_k
    n_std = (n - mu_n) / std_n

    # f(x, y) — raw field, then globally standardised
    x = np.linspace(0, 1, grid_size, dtype=np.float32)
    y = np.linspace(0, 1, grid_size, dtype=np.float32)
    x, y = np.meshgrid(x, y)
    f_xy = A * (mu1**2 + mu2**2 + x**2 + y**2) * np.sin(mu1 * np.pi * x) * np.sin(mu2 * np.pi * y)
    f_std = (f_xy - mu_f) / std_f

    # BC mask — zero interior, bc_value on active boundaries
    bc_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    if 1 in boundary_conditions:  # left
        bc_mask[:, 0]  = bc_value
    if 2 in boundary_conditions:  # right
        bc_mask[:, -1] = bc_value
    if 3 in boundary_conditions:  # bottom
        bc_mask[0, :]  = bc_value
    if 4 in boundary_conditions:  # top
        bc_mask[-1, :] = bc_value

    channels = [
        np.full((grid_size, grid_size), k_std, dtype=np.float32),  # ch 0: k standardised
        np.full((grid_size, grid_size), n_std, dtype=np.float32),  # ch 1: n standardised
        f_std.astype(np.float32),                                    # ch 2: f globally standardised
        bc_mask,                                                      # ch 3: BC mask
    ]
    return np.stack(channels, axis=0)  # [4, H, W]


def sample_param_sets(alpha_array, num_samples, stratify=True, rng=None):
    """Sample rows from alpha_array (columns: k, A, n, mu1, mu2) and return
    a list of param_set dicts.
    
    stratify=False: uniform random sample without replacement.
    stratify=True:  equal samples per unique n value (num_samples must be
                    divisible by the number of unique n values).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if stratify:
        n_vals = np.unique(alpha_array[:, 2])
        per_n = num_samples // len(n_vals)
        idx = np.concatenate([
            rng.choice(np.where(alpha_array[:, 2] == n)[0], per_n, replace=False)
            for n in n_vals
        ])
    else:
        idx = rng.choice(len(alpha_array), num_samples, replace=False)
    
    return [
        {"k": float(r[0]), "A": float(r[1]), "n": int(r[2]),
         "mu1": float(r[3]), "mu2": float(r[4])}
        for r in alpha_array[idx]
    ]



def utils_save_dataset(generate_sample_func, num_samples, seed, dataset_type, config, training_type):
    """dataset_type in {'train', 'valid', 'test'}. Train has no solution_tensor."""
    n_pts = config['data_config']['n_pts']
    safe_sample_set = np.load(config['data_config']['safe_sample_set'])

    key = {'train': 'train_path',
           'valid': 'val_path',
           'test':  'test_path'}[dataset_type]
    num_samples = num_samples
    save_path   = config['data_config'][key]

    if training_type == 'datadriven':
        save_solution = True
    else:
        save_solution = (dataset_type != 'train')
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    rng = np.random.default_rng(seed)
    shuffled = safe_sample_set[rng.permutation(len(safe_sample_set))]
    if num_samples > len(shuffled):
        raise ValueError(f"Requested {num_samples} > {len(shuffled)} available.")

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('k',   (num_samples,), dtype=np.float64)
        f.create_dataset('n',   (num_samples,), dtype=np.int32)
        f.create_dataset('A',   (num_samples,), dtype=np.float64)
        f.create_dataset('mu1', (num_samples,), dtype=np.float64)
        f.create_dataset('mu2', (num_samples,), dtype=np.float64)
        f.create_dataset('boundary_conditions', (num_samples,), 
                         dtype=h5py.special_dtype(vlen=np.int32))
        f.create_dataset('dirichlet_value', (num_samples,), dtype=np.float64)
        f.create_dataset('input_tensor', (num_samples, 4, n_pts, n_pts), dtype=np.float64)
        f.create_dataset('sample_index', (num_samples,), dtype=np.int32)
        if save_solution:
            f.create_dataset('solution_tensor', (num_samples, 1, n_pts, n_pts), 
                             dtype=np.float64)

        csv_path = save_path.replace('.h5', '_solver_stats.csv')
        csv_fields = ['sample_index', 'k', 'n', 'A', 'mu1', 'mu2',
                      'dirichlet_value', 'solver_iters', 'cg_iters']
        csv_file = open(csv_path, 'w', newline='') if save_solution else None
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields) if csv_file else None
        if csv_writer:
            csv_writer.writeheader()

        for i in range(num_samples):
            row = shuffled[i]
            param_set = {'k': float(row[0]), 'A': float(row[1]), 'n': int(row[2]),
                         'mu1': float(row[3]), 'mu2': float(row[4])}
            print(f'Generating sample {i}/{num_samples}')
            
            out = generate_sample_func(i, seed, dataset_type=dataset_type,
                                       param_set=param_set, config=config,
                                       training_type=training_type)
            if save_solution:
                input_tensor, params_dict, solution_tensor, iters, cg_iters = out
                f['solution_tensor'][i] = solution_tensor
                csv_writer.writerow({
                    'sample_index': i,
                    'k':   np.round(params_dict['k'],   decimals=3),
                    'n':   params_dict['n'],
                    'A':   np.round(params_dict['A'],   decimals=3),
                    'mu1': np.round(params_dict['mu1'], decimals=3),
                    'mu2': np.round(params_dict['mu2'], decimals=3),
                    'dirichlet_value': np.round(params_dict['dirichlet_value'], decimals=3),
                    'solver_iters': iters,
                    'cg_iters': cg_iters,
                })
                csv_file.flush()
            else:
                input_tensor, params_dict = out

            f['sample_index'][i]        = i
            f['input_tensor'][i]        = input_tensor
            f['k'][i]                   = np.round(params_dict['k'],   decimals=3)
            f['n'][i]                   = params_dict['n']
            f['A'][i]                   = np.round(params_dict['A'],   decimals=3)
            f['mu1'][i]                 = np.round(params_dict['mu1'], decimals=3)
            f['mu2'][i]                 = np.round(params_dict['mu2'], decimals=3)
            f['boundary_conditions'][i] = params_dict['boundary_conditions']
            f['dirichlet_value'][i]     = np.round(params_dict['dirichlet_value'], decimals=3)

        if csv_file:
            csv_file.close()
            print(f'Solver stats saved to {csv_path}')

    print(f'Saved {num_samples} to {save_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Train PINN model for 3D elasticity problem using YAML config')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    return parser.parse_args()

def main(config_path): 

    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    n_pts = config['data_config']['n_pts']
    x_pts, y_pts, xv, yv, nx, ny = sampling_points_2d_grid(n_pts=n_pts)

    # split into train, test, valid
    split_paths, n_train, n_val, n_test = split_safe_set(
        safe_set_path=config['data_config']['safe_sample_set'],
        out_dir=config['data_config']['split_dir'],
        total_num_samples=config['data_config']['total_num_samples'],
        train_frac=config['data_config'].get('train_frac'),
        val_frac=config['data_config'].get('val_frac'),
        seed=config['data_config'].get('split_seed'),
    )

    training_type = config['data_config']['training_type'] # data driven or solver guided

    # Compute global f(x,y) stats from the training split before generating any data.
    grid_size = config['data_config']['n_pts']
    train_param_rows = [
        {'k': float(r[0]), 'A': float(r[1]), 'n': int(r[2]), 'mu1': float(r[3]), 'mu2': float(r[4])}
        for r in np.load(split_paths['train'])
    ]
    print(f'Computing global standardization stats over {len(train_param_rows)} training samples...')
    mu_k, std_k, mu_n, std_n, mu_f, std_f = compute_fxy_stats(train_param_rows, grid_size, config)
    print(f'  mu_k={mu_k:.6f}  std_k={std_k:.6f}')
    print(f'  mu_n={mu_n:.6f}  std_n={std_n:.6f}')
    print(f'  mu_f={mu_f:.6f}  std_f={std_f:.6f}')
    config['data_config']['k_mu']   = mu_k
    config['data_config']['k_std']  = std_k
    config['data_config']['n_mu']   = mu_n
    config['data_config']['n_std']  = std_n
    config['data_config']['fxy_mu'] = mu_f
    config['data_config']['fxy_std'] = std_f

    # Persist stats so inference / training scripts can reload them without re-scanning.
    stats_path = os.path.join(config['data_config']['split_dir'], 'fxy_stats.npz')
    np.savez(stats_path,
             mu_k=np.array(mu_k), std_k=np.array(std_k),
             mu_n=np.array(mu_n), std_n=np.array(std_n),
             mu_f=np.array(mu_f), std_f=np.array(std_f))
    print(f'  Stats saved to {stats_path}')

    # Generate each dataset from its own disjoint pool.
    for dataset_type, num_samples, seed in [('train', n_train, 24), ('valid', n_val, 129), ('test', n_test, 730)]:
        print(f"\n=== Generating {dataset_type} dataset ===")
        # Override the safe_sample_set path to point at the correct split.
        config['data_config']['safe_sample_set'] = split_paths[dataset_type]
        
        if dataset_type == 'train':
            utils_save_dataset(
                generate_sample_func=utils_generate_sample,
                num_samples=num_samples, 
                seed=seed,
                dataset_type=dataset_type,
                config=config,
                training_type=training_type,
            )
        else:
            utils_save_dataset(
                generate_sample_func=utils_generate_sample,
                num_samples=num_samples, 
                seed=seed,
                dataset_type=dataset_type,
                config=config,
                training_type=training_type,
            )

    print("\nAll datasets generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to dataset generation YAML config.")
    args = parser.parse_args()
    main(args.config)

     

