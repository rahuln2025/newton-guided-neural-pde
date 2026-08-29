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
from utils_train_with_solver import *
from utils_plots import plot_residuals_comparison
import matplotlib

import h5py
import yaml
import argparse
from torchinfo import summary


def inference_and_plot(model, dataset, sample_indices, x_pts, y_pts, config, plot_dir, device):
    """
    Perform batch inference and plotting for multiple samples.
    
    Args:
        model: Trained model
        dataset: PDEDataset instance
        sample_indices: List of sample indices to evaluate
        x_pts, y_pts: Grid points for solver
        config: Configuration dictionary
        plot_dir: Directory to save plots
        device: torch device
    """
    import csv
    import sys
    
    model.eval()
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create log file to capture all print outputs
    log_file_path = os.path.join(plot_dir, 'inference_log.txt')
    log_file = open(log_file_path, 'w')
    
    # Create a custom print function that writes to both console and log file
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file)
        log_file.flush()
    
    # Batch load all samples
    samples_data = []
    for idx in sample_indices:
        input_tensor, alpha, f, mu1, mu2, n, bcs, bc_values, sample_idx = dataset[idx]
        samples_data.append({
            'input_tensor': input_tensor,
            'alpha': alpha,
            'f': f,
            'mu1': mu1,
            'mu2': mu2,
            'n': n,
            'bcs': bcs,
            'bc_values': bc_values,
            'original_sample_idx': sample_idx,
            'idx': idx
        })
    
    # Batch prepare inputs
    prec = get_precision_config(config)
    input_batch = torch.stack([torch.from_numpy(s['input_tensor']) for s in samples_data]).to(device=device, dtype=prec['input_dtype'])
    
    # Batch inference
    with torch.no_grad():
        pred_batch = model(input_batch)
    
    pred_batch_np = pred_batch.squeeze(1).detach().to(dtype=prec['solver_output_torch_dtype']).cpu().numpy()
    
    maxiters = 100
    N_REPEATS = 3  # median over this many timed repetitions per solve

    # CG solver config
    _solver_config = config.get('solver_config', {})
    _linear_solver = _solver_config.get('linear_solver')
    _cg_maxiter = int(_solver_config.get('cg_maxiter'))
    _cg_tol = float(_solver_config.get('cg_tol'))
    _dampfactor = float(_solver_config.get('newton_dampfactor', 1.0))
    
    # Grid for plotting
    xv, yv = np.meshgrid(x_pts, y_pts, indexing='ij')
    
    # ── Burn-in: one untimed solve to prime NGSolve mesh cache + OS allocator ──
    _s0 = samples_data[0]
    solve_pde(u_init=None, k=_s0['alpha'], n=int(_s0['n']), A=_s0['f'],
              mu1=_s0['mu1'], mu2=_s0['mu2'],
              dirichlet_bcs=_s0['bcs'].tolist() if isinstance(_s0['bcs'], np.ndarray) else _s0['bcs'],
              dirichlet_value=_s0['bc_values'], x_pts=x_pts, y_pts=y_pts,
              maxiters=maxiters, verbose=False,
              linear_solver=_linear_solver, cg_maxiter=_cg_maxiter, cg_tol=_cg_tol,
              dampfactor=_dampfactor)

    # Randomize per-sample order to eliminate systematic cold-first cache bias.
    # Half the samples are timed cold-first, half warm-first.
    rng = np.random.default_rng(seed=42)
    cold_first_mask = np.array([True] * (len(samples_data) // 2) +
                                [False] * (len(samples_data) - len(samples_data) // 2))
    rng.shuffle(cold_first_mask)

    # Process each sample
    results = []
    for i, sample in enumerate(samples_data):
        idx = sample['idx']
        alpha = sample['alpha']
        f_param = sample['f']
        mu1 = sample['mu1']
        mu2 = sample['mu2']
        n = sample['n']
        bcs = sample['bcs']
        bc_values = sample['bc_values']
        sample_idx = sample['original_sample_idx']
        
        pred = pred_batch_np[i]
        
        log_print(f"\nSample {sample_idx}: alpha={alpha:.2f}, n={n}, f={f_param:.2f}, mu1={mu1:.2f}, mu2={mu2:.2f}")
        log_print(f"BCs: {bcs}, BC values: {bc_values}")
        
        # Plot prediction
        plot_2d_scatter_tensor(xv, yv, idx, plot_dir, 
                              pred[np.newaxis, :, :], 
                              savefile='pred')
        
        # ── Solve args (shared) ───────────────────────────────────────────
        _bcs = bcs.tolist() if isinstance(bcs, np.ndarray) else bcs
        _solve_kwargs = dict(k=alpha, n=int(n), A=f_param, mu1=mu1, mu2=mu2,
                             dirichlet_bcs=_bcs, dirichlet_value=bc_values,
                             x_pts=x_pts, y_pts=y_pts, maxiters=maxiters,
                             verbose=False, linear_solver=_linear_solver,
                             cg_maxiter=_cg_maxiter, cg_tol=_cg_tol,
                             dampfactor=_dampfactor)

        pred_warm = apply_bc_mask_np(pred, _bcs,
                                     bc_values if np.isscalar(bc_values) else float(bc_values),
                                     len(x_pts))

        def _time_solve(u_init):
            times = []
            for _ in range(N_REPEATS):
                t0 = time.perf_counter()
                out = solve_pde(u_init=u_init, **_solve_kwargs)
                times.append(time.perf_counter() - t0)
            return out, float(np.median(times))

        # Randomized order eliminates systematic cache-warming bias
        run_cold_first = cold_first_mask[i]
        if run_cold_first:
            (u_sol, iters, _, cg_iters_cold, _, residuals_cold), t_cold = _time_solve(None)
            (u_pred_refined, iters_refined, _, cg_iters_warm, _, residuals_warm), t_warm = _time_solve(pred_warm)
        else:
            (u_pred_refined, iters_refined, _, cg_iters_warm, _, residuals_warm), t_warm = _time_solve(pred_warm)
            (u_sol, iters, _, cg_iters_cold, _, residuals_cold), t_cold = _time_solve(None)

        log_print(f"Ground truth solver iterations: {iters}  (median {t_cold:.3f}s over {N_REPEATS} runs, {'cold-first' if run_cold_first else 'warm-first'})")
        log_print(f"CG iters per Newton step (cold): {cg_iters_cold}")
        
        # Plot ground truth
        plot_2d_scatter_tensor(xv, yv, idx, plot_dir, u_sol, savefile='truth')
        
        # Compute and plot difference
        diff = np.abs(pred - u_sol[0])
        l2_error = np.linalg.norm(diff)
        mse_error = np.mean((pred - u_sol[0])**2)
        log_print(f"L2 error (pred vs truth): {l2_error:.6e}")
        log_print(f"MSE (pred vs truth): {mse_error:.6e}")
        plot_2d_scatter_tensor(xv, yv, idx, plot_dir, 
                              diff[np.newaxis, :, :], 
                              savefile='diff', 
                              cmap='coolwarm')
        
        time_speedup_pct = (1.0 - t_warm / t_cold) * 100.0 if t_cold > 0 else 0.0
        log_print(f"Refined solver iterations: {iters_refined}  (median {t_warm:.3f}s)  time speedup: {time_speedup_pct:+.1f}%")
        log_print(f"CG iters per Newton step (warm): {cg_iters_warm}")
        cg_total_cold = int(sum(cg_iters_cold)) if isinstance(cg_iters_cold, list) else int(cg_iters_cold)
        cg_total_warm = int(sum(cg_iters_warm)) if isinstance(cg_iters_warm, list) else int(cg_iters_warm)
        log_print(f"Total CG iters — cold: {cg_total_cold}, warm: {cg_total_warm}  (reduction: {cg_total_cold - cg_total_warm:+d})")
        
        # Plot refined solution
        plot_2d_scatter_tensor(xv, yv, idx, plot_dir, u_pred_refined, savefile='refined')
        
        # Compute and plot refined difference (clip numerical noise below solver tolerance)
        diff_refined = np.where(np.abs(u_sol[0] - u_pred_refined[0]) < 1e-14, 0.0, np.abs(u_sol[0] - u_pred_refined[0]))
        l2_error_refined = np.linalg.norm(diff_refined)
        log_print(f"L2 error (refined vs truth): {l2_error_refined:.6e}")
        plot_2d_scatter_tensor(xv, yv, idx, plot_dir, 
                              diff_refined[np.newaxis, :, :], 
                              savefile='diff_refined', 
                              cmap='coolwarm')
        
        results.append({
            'idx': sample_idx,
            'l2_error': l2_error,
            'mse_error': mse_error,
            'l2_error_refined': l2_error_refined,
            'solver_iters': iters,
            'refined_iters': iters_refined,
            'speedup': iters / iters_refined if iters_refined > 0 else float('inf'),
            't_cold_s': t_cold,
            't_warm_s': t_warm,
            'time_speedup_pct': time_speedup_pct,
            'cg_iters_cold': cg_iters_cold,
            'cg_iters_warm': cg_iters_warm,
            'cg_total_cold': cg_total_cold,
            'cg_total_warm': cg_total_warm,
            'residuals': residuals_cold,
            'residuals_refined': residuals_warm,
        })
    
    # Print summary
    log_print("\n" + "="*80)
    log_print("INFERENCE SUMMARY")
    log_print(f"Timing: median of {N_REPEATS} repetitions per solve; order randomized per sample (burn-in applied)")
    log_print("="*80)
    log_print(f"{'Idx':<6} {'MSE':<12} {'L2 Error':<12} {'L2 Refined':<12} {'Iters':<8} {'Iters W':<8} {'Iter Spdup':<11} {'t_cold(s)':<11} {'t_warm(s)':<11} {'Time Spdup%':<12} {'CG cold':<9} {'CG warm':<9}")
    log_print("-"*115)
    for r in results:
        log_print(f"{r['idx']:<6} {r['mse_error']:<12.6e} {r['l2_error']:<12.6e} {r['l2_error_refined']:<12.6e} "
              f"{r['solver_iters']:<8} {r['refined_iters']:<8} {r['speedup']:<11.2f} "
              f"{r['t_cold_s']:<11.3f} {r['t_warm_s']:<11.3f} {r['time_speedup_pct']:<+12.1f}"
              f"{r['cg_total_cold']:<9} {r['cg_total_warm']:<9}")
    log_print("="*115)

    avg_mse = np.mean([r['mse_error'] for r in results])
    avg_speedup = np.mean([r['speedup'] for r in results if r['speedup'] != float('inf')])
    avg_t_cold = np.mean([r['t_cold_s'] for r in results])
    avg_t_warm = np.mean([r['t_warm_s'] for r in results])
    avg_time_speedup_pct = np.mean([r['time_speedup_pct'] for r in results])
    avg_cg_cold = np.mean([r['cg_total_cold'] for r in results])
    avg_cg_warm = np.mean([r['cg_total_warm'] for r in results])
    log_print(f"Average MSE: {avg_mse:.6e}")
    log_print(f"Average iter speedup: {avg_speedup:.2f}x")
    log_print(f"Average cold-start time: {avg_t_cold:.3f}s  |  warm-start: {avg_t_warm:.3f}s  |  time speedup: {avg_time_speedup_pct:+.1f}%")
    log_print(f"Average total CG iters — cold: {avg_cg_cold:.1f}, warm: {avg_cg_warm:.1f}  (avg reduction: {avg_cg_cold - avg_cg_warm:+.1f})")

    # Save results to CSV file
    csv_file_path = os.path.join(plot_dir, 'inference_results.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['idx', 'mse_error', 'l2_error', 'l2_error_refined',
                      'solver_iters', 'refined_iters', 'speedup',
                      't_cold_s', 't_warm_s', 'time_speedup_pct',
                      'cg_total_cold', 'cg_total_warm',
                      'cg_iters_cold', 'cg_iters_warm',
                      'residuals', 'residuals_refined']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')

        writer.writeheader()
        for r in results:
            row = dict(r)
            row['residuals'] = ';'.join(f'{v:.6e}' for v in r['residuals']) if r['residuals'] else ''
            row['residuals_refined'] = ';'.join(f'{v:.6e}' for v in r['residuals_refined']) if r['residuals_refined'] else ''
            writer.writerow(row)

        # Add summary row
        writer.writerow({
            'idx': 'AVERAGE',
            'mse_error': avg_mse,
            'l2_error': '',
            'l2_error_refined': '',
            'solver_iters': '',
            'refined_iters': '',
            'speedup': avg_speedup,
            't_cold_s': avg_t_cold,
            't_warm_s': avg_t_warm,
            'time_speedup_pct': avg_time_speedup_pct,
            'cg_total_cold': avg_cg_cold,
            'cg_total_warm': avg_cg_warm,
            'cg_iters_cold': '',
            'cg_iters_warm': '',
        })
    
    log_print(f"\nResults saved to: {csv_file_path}")
    log_print(f"Log saved to: {log_file_path}")
    
    # Close log file
    log_file.close()

    # plot residuals vs iterations for all samples
    plot_residuals_comparison(results, plot_dir)
    
    return results


def sweep_checkpoints(ckpt_dir, ckpt_epochs, config, dataset, x_pts, y_pts,
                      sample_indices, device, base_plot_dir):
    """
    Run inference_and_plot for each checkpoint epoch and collect a summary table.

    Args:
        ckpt_dir       : directory containing ckpt_<epoch>.pt files
        ckpt_epochs    : list of epoch integers to evaluate
        config         : loaded config dict
        dataset        : PDEDataset instance (sample_type='train')
        x_pts, y_pts   : grid points
        sample_indices : sample indices passed to inference_and_plot
        device         : torch device
        base_plot_dir  : parent dir; per-epoch results go in <base_plot_dir>/epoch_<ep>/

    Saves:
        <base_plot_dir>/sweep_summary.csv
    Prints:
        Epoch | Avg MSE | Avg Speedup table
    """
    import csv

    precision_config = get_precision_config(config)
    summary_rows = []

    for ep in ckpt_epochs:
        ckpt_path = os.path.join(ckpt_dir, f'ckpt_{ep}.pt')
        if not os.path.exists(ckpt_path):
            print(f'[sweep] Checkpoint not found, skipping: {ckpt_path}')
            continue

        print(f'\n{"="*60}')
        print(f'[sweep] Epoch {ep}: {ckpt_path}')
        print(f'{"="*60}')

        model = create_model(config)
        model = convert_model_precision(model, precision_config['model_dtype'])
        model = model.to(device=device)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        state_dict.pop('_metadata', None)
        model.load_state_dict(state_dict)

        plot_dir = os.path.join(base_plot_dir, f'epoch_{ep}')
        os.makedirs(plot_dir, exist_ok=True)

        results = inference_and_plot(model, dataset, sample_indices,
                                     x_pts, y_pts, config, plot_dir, device)

        valid_speedups = [r['speedup'] for r in results if r['speedup'] != float('inf')]
        avg_mse     = float(np.mean([r['mse_error'] for r in results]))
        avg_speedup = float(np.mean(valid_speedups)) if valid_speedups else float('inf')

        summary_rows.append({'epoch': ep, 'avg_mse': avg_mse, 'avg_speedup': avg_speedup})
        print(f'[sweep] ep={ep:>5}  avg_mse={avg_mse:.4e}  avg_speedup={avg_speedup:.2f}x')

        del model

    # Print table
    print(f'\n{"="*50}')
    print(f'{"Epoch":>8}  {"Avg MSE":>14}  {"Avg Speedup":>12}')
    print(f'{"-"*50}')
    for row in summary_rows:
        print(f'{row["epoch"]:>8}  {row["avg_mse"]:>14.4e}  {row["avg_speedup"]:>11.2f}x')
    print(f'{"="*50}')

    # Save CSV
    os.makedirs(base_plot_dir, exist_ok=True)
    csv_path = os.path.join(base_plot_dir, 'sweep_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'avg_mse', 'avg_speedup'])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'\nSummary saved to: {csv_path}')

    return summary_rows


def _parse_args():
    parser = argparse.ArgumentParser(description='Inference for 2D PDE neural operators')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--gpu',    type=int, default=0,    help='GPU device ID')
    parser.add_argument('--ckpt',   type=str, default=None,
                        help='Override checkpoint path (default: uses inference_config.checkpoint)')
    parser.add_argument('--sample_indices', '--samples_indices', type=int, nargs='+', default=None,
                        help='Override sample indices from config (e.g. --sample_indices 0 1 2 5)')
    return parser.parse_args()


def main():
    args = _parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = resolve_run_paths(config)

    # ── Inference config block ─────────────────────────────────────────────
    ic          = config.get('inference_config', {})
    run_name    = config.get('run_name', 'run')
    run_root    = config.get('run_root', './runs')
    ckpt_dir    = config['training_config']['checkpoint_dir']

    # Checkpoint: CLI --ckpt > config checkpoint > best_model.pt
    ckpt_override = args.ckpt or ic.get('checkpoint')
    load_checkpoint_path = (
        ckpt_override if ckpt_override
        else os.path.join(ckpt_dir, 'best_model.pt')
    )

    run_sweep      = ic.get('run_sweep',      True)
    run_single     = ic.get('run_single',     False)
    sweep_epochs   = ic.get('sweep_epochs',   [])
    dataset_type   = ic.get('dataset',        'test')
    sample_indices = np.array(args.sample_indices if args.sample_indices is not None
                              else ic.get('sample_indices', list(range(32))))

    # All output paths derive from run_name
    plot_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    infer_base     = os.path.join(run_root, run_name, 'inference')
    plot_dir       = os.path.join(infer_base, f'single_{plot_timestamp}')
    sweep_plot_dir = os.path.join(infer_base, f'sweep_{plot_timestamp}')
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n=== Inference  |  run: {run_name} ===")
    print(f"  checkpoint_dir : {ckpt_dir}")
    print(f"  load_ckpt      : {load_checkpoint_path}")
    print(f"  output root    : {infer_base}")
    print(f"  sweep epochs   : {sweep_epochs}")
    print(f"  samples        : {sample_indices}\n")

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    # ── Grid & datasets ───────────────────────────────────────────────────
    n_pts = config['data_config']['n_pts']
    x_pts, y_pts, xv, yv, nx, ny = sampling_points_2d_grid(n_pts=n_pts)

    train_dataset = PDEDataset(config['data_config']['train_path'], config, sample_type='train')
    test_dataset  = PDEDataset(config['data_config']['test_path'],  config, sample_type='train')
    dataset = train_dataset if dataset_type == 'train' else test_dataset

    # ── Model ─────────────────────────────────────────────────────────────
    precision_config = get_precision_config(config)
    model = create_model(config)
    model = convert_model_precision(model, precision_config['model_dtype'])
    model = model.to(device=device)

    if os.path.exists(load_checkpoint_path):
        checkpoint = torch.load(load_checkpoint_path, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        state_dict.pop('_metadata', None)
        model.load_state_dict(state_dict)
        print(f"Loaded: {load_checkpoint_path}")
        print(f"  epoch: {checkpoint.get('epoch', 'N/A')}  best_loss: {checkpoint.get('best_loss', 'N/A')}")
    else:
        print(f"Checkpoint not found: {load_checkpoint_path}")
        return

    # ── Sweep ──────────────────────────────────────────────────────────────
    if run_sweep:
        sweep_checkpoints(
            ckpt_dir=ckpt_dir,
            ckpt_epochs=sweep_epochs,
            config=config,
            dataset=dataset,
            x_pts=x_pts,
            y_pts=y_pts,
            sample_indices=sample_indices,
            device=device,
            base_plot_dir=sweep_plot_dir,
        )

    # ── Single ─────────────────────────────────────────────────────────────
    if run_single:
        inference_results = inference_and_plot(
            model, dataset, sample_indices,
            x_pts, y_pts, config, plot_dir, device,
        )
        print(inference_results)


if __name__ == '__main__':
    main()
