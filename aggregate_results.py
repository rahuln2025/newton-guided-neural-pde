"""
Aggregate inference_results.csv results across all seed replicates.
Usage:
    python aggregate_results.py --runs_dir ./runs --output results/doe_summary.csv
"""
import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots


# Maps run_name patterns to DoE factors
_RUN_PATTERN = re.compile(
    r'(?P<size>\d+)_UNet2D_(?P<variant>dd|solver_M\d+)(?:_seed(?P<seed>\d+))?'
)


def parse_run_name(run_name: str) -> dict:
    m = _RUN_PATTERN.search(run_name)
    if not m:
        return None
    seed = int(m.group('seed')) if m.group('seed') else 0
    variant = m.group('variant')
    maxiters = None
    if variant.startswith('solver_M'):
        maxiters = int(variant.replace('solver_M', ''))
        variant = 'solver'
    return {
        'dataset_size': int(m.group('size')),
        'variant': variant,
        'maxiters': maxiters,
        'seed': seed,
    }


# Matches inference output dirs: test_id_seed0, test_ood_seed1, etc.
_TEST_DIR_PATTERN = re.compile(r'^test_(?P<split>id|ood)_seed(?P<test_seed>\d+)$')


def _load_csv(csv_path: str, min_solver_iters: int | None = None) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    df = df[df['idx'] != 'AVERAGE'].copy()
    if min_solver_iters is not None:
        df = df[df['solver_iters'] >= min_solver_iters].copy()
        if len(df) == 0:
            return None
    for col in ['mse_error', 'speedup', 'time_speedup_pct', 't_cold_s', 't_warm_s',
                'cg_total_cold', 'cg_total_warm', 'solver_iters', 'refined_iters']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'r_cg_per_step' in df.columns:
        df['r_cg_per_step'] = pd.to_numeric(df['r_cg_per_step'], errors='coerce')
    else:
        cbar_cold = df['cg_total_cold'] / df['solver_iters']
        cbar_warm = df['cg_total_warm'] / df['refined_iters']
        df['r_cg_per_step'] = cbar_cold / cbar_warm
    return df


def find_inference_csvs(runs_dir: str, min_solver_iters: int | None = None) -> list[dict]:
    """Return one record per (run_name, test_split, test_seed) triple."""
    records = []
    for run_name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        factors = parse_run_name(run_name)
        if factors is None:
            continue

        inference_dir = os.path.join(run_dir, 'inference')
        if not os.path.isdir(inference_dir):
            continue

        for entry in sorted(os.listdir(inference_dir)):
            m = _TEST_DIR_PATTERN.match(entry)
            if m is None:
                continue  # skip single_<timestamp> and sweep dirs
            test_split = m.group('split')          # 'id' or 'ood'
            test_seed  = int(m.group('test_seed'))

            csv_path = os.path.join(inference_dir, entry, 'inference_results.csv')
            if not os.path.isfile(csv_path):
                continue

            df = _load_csv(csv_path, min_solver_iters=min_solver_iters)
            if df is None or len(df) == 0:
                continue

            records.append({
                **factors,
                'run_name':          run_name,
                'test_split':        test_split,
                'test_seed':         test_seed,
                'n_samples':         len(df),
                'avg_mse':           df['mse_error'].mean(),
                'std_mse':           df['mse_error'].std(),
                'avg_R':             df['speedup'].mean(),
                'std_R':             df['speedup'].std(),
                'avg_R_cg_per_step': df['r_cg_per_step'].mean(),
                'std_R_cg_per_step': df['r_cg_per_step'].std(),
                'avg_St_pct':        df['time_speedup_pct'].mean(),
                'std_St_pct':        df['time_speedup_pct'].std(),
                'avg_t_cold_s':      df['t_cold_s'].mean(),
                'avg_t_warm_s':      df['t_warm_s'].mean(),
                'csv_path':          csv_path,
            })
    return records


def aggregate(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (agg_id, agg_ood) — one aggregated table per test split.

    Each table averages over all (model_seed, test_seed) pairs, giving
    n_pairs = n_model_seeds x n_test_seeds entries per condition.
    Expected: 3 model seeds x 3 ID test seeds = 9 pairs for ID;
              3 model seeds x 2 OOD test seeds = 6 pairs for OOD.
    """
    df = pd.DataFrame(records)
    group_cols = ['dataset_size', 'variant', 'maxiters']

    def _agg_split(sub: pd.DataFrame) -> pd.DataFrame:
        return (sub.groupby(group_cols, dropna=False)
                   .agg(
                       n_pairs=('seed', 'count'),
                       n_model_seeds=('seed', 'nunique'),
                       n_test_seeds=('test_seed', 'nunique'),
                       mse_mean=('avg_mse', 'mean'),
                       mse_std=('avg_mse', 'std'),
                       R_mean=('avg_R', 'mean'),
                       R_std=('avg_R', 'std'),
                       R_cg_mean=('avg_R_cg_per_step', 'mean'),
                       R_cg_std=('avg_R_cg_per_step', 'std'),
                       St_mean=('avg_St_pct', 'mean'),
                       St_std=('avg_St_pct', 'std'),
                       t_cold_mean=('avg_t_cold_s', 'mean'),
                       t_warm_mean=('avg_t_warm_s', 'mean'),
                   )
                   .reset_index()
                   .sort_values(['dataset_size', 'variant', 'maxiters']))

    agg_id  = _agg_split(df[df['test_split'] == 'id'])
    agg_ood = _agg_split(df[df['test_split'] == 'ood'])
    return agg_id, agg_ood


def plot_doe_summary(agg_df: pd.DataFrame, plot_dir: str) -> None:
    plt.style.use(['science', 'grid', 'no-latex'])
    plt.rcParams.update({
        'figure.dpi':               300,
        'savefig.dpi':              300,
        'figure.titlesize':         18,
        'xtick.labelsize':          12,
        'ytick.labelsize':          12,
        'axes.labelsize':           16,
        'axes.titlesize':           16,
        'axes.formatter.use_mathtext': True,
        'font.family':              'DejaVu Sans',
    })
    os.makedirs(plot_dir, exist_ok=True)

    # ── layout ────────────────────────────────────────────────────────────
    sizes      = sorted(agg_df['dataset_size'].unique())
    colors     = {128: '#0077BB', 256: '#EE7733', 512: '#009988'}

    # x-axis: solver_M1 / M3 / M5 first, then dd
    def _sort_key(row):
        if row['variant'] == 'dd':
            return (1, 0)
        return (0, row['maxiters'])

    agg_sorted = agg_df.copy()
    agg_sorted['_k'] = agg_sorted.apply(_sort_key, axis=1)
    agg_sorted = agg_sorted.sort_values(['dataset_size', '_k']).drop(columns='_k')

    # one canonical x-label order (from first size)
    label_rows  = agg_sorted[agg_sorted['dataset_size'] == sizes[0]]
    x_labels    = []
    for _, r in label_rows.iterrows():
        x_labels.append('DD' if r['variant'] == 'dd' else f"M={int(r['maxiters'])}")
    n_models    = len(x_labels)
    x_ticks     = np.arange(n_models)

    n_sizes  = len(sizes)
    group_w  = 0.6
    bar_w    = group_w / n_sizes
    offsets  = np.linspace(-group_w / 2 + bar_w / 2, group_w / 2 - bar_w / 2, n_sizes)

    plots = [
        ('mse_mean',  'mse_std',  'MSE',
         'log',    'plots/doe_mse_vs_budget.pdf'),
        ('R_mean',    'R_std',    r'$R$',
         'linear', 'plots/doe_R_vs_budget.pdf'),
        ('R_cg_mean', 'R_cg_std',
         r'$R_\mathrm{CG}^\mathrm{per\,step}$',
         'linear', 'plots/doe_Rcg_vs_budget.pdf'),
        ('St_mean',   'St_std',
         r'$S_t\;(\%)$',
         'linear', 'plots/doe_St_vs_budget.pdf'),
    ]

    for mean_col, std_col, ylabel, yscale, fname in plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        handles = []
        for size, offset in zip(sizes, offsets):
            sub = agg_sorted[agg_sorted['dataset_size'] == size]
            means = sub[mean_col].values
            stds  = sub[std_col].fillna(0).values
            positions = x_ticks + offset
            ax.bar(positions, means, width=bar_w * 0.85,
                   color=colors[size], alpha=0.80,
                   yerr=stds,
                   error_kw=dict(elinewidth=1.0, capsize=3, ecolor='black'))
            handles.append(plt.Rectangle((0, 0), 1, 1,
                                         fc=colors[size], alpha=0.80,
                                         label=f'N={size}'))
        if yscale == 'log':
            ax.set_yscale('log')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Solver budget')
        ax.set_ylabel(ylabel)
        ax.legend(handles=handles, fontsize=10)
        fig.tight_layout()
        out = os.path.join(plot_dir, os.path.basename(fname))
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_dir', default='./runs')
    parser.add_argument('--output',   default='results/doe_summary.csv')
    parser.add_argument('--plot_dir', default='plots/doe')
    parser.add_argument('--min_solver_iters', type=int, default=None,
                        help='Keep only samples with solver_iters >= this value (e.g. 10 for nonlinear-only).')
    args = parser.parse_args()

    records = find_inference_csvs(args.runs_dir, min_solver_iters=args.min_solver_iters)
    if args.min_solver_iters is not None:
        print(f'[filter] keeping samples with solver_iters >= {args.min_solver_iters}')
    if not records:
        print('No inference_results.csv files found.')
        return

    raw_df = pd.DataFrame(records)
    agg_id, agg_ood = aggregate(records)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    raw_path     = args.output.replace('.csv', '_raw.csv')
    id_out_path  = args.output.replace('.csv', '_id.csv')
    ood_out_path = args.output.replace('.csv', '_ood.csv')

    raw_df.to_csv(raw_path,     index=False)
    agg_id.to_csv(id_out_path,  index=False)
    agg_ood.to_csv(ood_out_path, index=False)

    # Coverage report: expected 15 pairs per condition (3 model x 3 ID + 3 model x 2 OOD)
    print(f'\nRaw records: {len(raw_df)}  '
          f'(id={len(raw_df[raw_df["test_split"]=="id"])}, '
          f'ood={len(raw_df[raw_df["test_split"]=="ood"])}')
    print(f'Aggregated ID  → {id_out_path}  ({len(agg_id)} conditions)')
    print(f'Aggregated OOD → {ood_out_path}  ({len(agg_ood)} conditions)\n')

    print('── ID results ──')
    print(agg_id.to_string(index=False))
    print('\n── OOD results ──')
    print(agg_ood.to_string(index=False))

    print(f'\nGenerating plots → {args.plot_dir}/')
    plot_doe_summary(agg_id,  os.path.join(args.plot_dir, 'id'))
    plot_doe_summary(agg_ood, os.path.join(args.plot_dir, 'ood'))


if __name__ == '__main__':
    main()

