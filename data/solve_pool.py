"""
Run the NGSolve solver over every row in a parameter CSV and record
Newton iterations, per-step CG iterations, total CG iterations,
final residual, and convergence status.

No .h5 output — solver diagnostics only.

Usage:
    python data/solve_pool.py \
        --params_csv data/test_sets/test_params_pool.csv \
        --output     data/test_sets/pool_solver_stats.csv \
        --n_pts      64 \
        --maxiters   100
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import solve_pde, sampling_points_2d_grid

DIRICHLET_BCS   = [1, 2, 3, 4]
DIRICHLET_VALUE = 0.0

FIELDS = [
    'idx', 'split', 'k', 'A', 'n', 'mu1', 'mu2',
    'converged', 'nr_iters', 'final_residual',
    'cg_iters_per_step', 'cg_total',
    'wall_s',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_csv', default='data/test_sets/test_params_pool.csv')
    parser.add_argument('--output',     default='data/test_sets/pool_solver_stats.csv')
    parser.add_argument('--n_pts',      type=int,   default=64)
    parser.add_argument('--maxiters',   type=int,   default=100)
    parser.add_argument('--linear_solver', default='cg')
    parser.add_argument('--cg_maxiter', type=int,   default=1000)
    parser.add_argument('--cg_tol',     type=float, default=1e-12)
    parser.add_argument('--dampfactor', type=float, default=1.0)
    args = parser.parse_args()

    params = pd.read_csv(args.params_csv)
    x_pts, y_pts, _, _, _, _ = sampling_points_2d_grid(n_pts=args.n_pts)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── burn-in ────────────────────────────────────────────────────────────
    r0 = params.iloc[0]
    solve_pde(u_init=None, k=float(r0['k']), n=int(r0['n']), A=float(r0['A']),
              mu1=float(r0['mu1']), mu2=float(r0['mu2']),
              dirichlet_bcs=DIRICHLET_BCS, dirichlet_value=DIRICHLET_VALUE,
              x_pts=x_pts, y_pts=y_pts, maxiters=args.maxiters, verbose=False,
              linear_solver=args.linear_solver, cg_maxiter=args.cg_maxiter,
              cg_tol=args.cg_tol, dampfactor=args.dampfactor)
    print(f"Burn-in done. Running {len(params)} samples...\n")

    n_converged, n_diverged = 0, 0

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for i, row in params.iterrows():
            k   = float(row['k'])
            A   = float(row['A'])
            nv  = int(row['n'])
            mu1 = float(row['mu1'])
            mu2 = float(row['mu2'])
            split = str(row.get('split', ''))

            t0 = time.perf_counter()
            try:
                u_sol, nr_iters, _, cg_iters, _, residuals = solve_pde(
                    u_init=None, k=k, n=nv, A=A, mu1=mu1, mu2=mu2,
                    dirichlet_bcs=DIRICHLET_BCS, dirichlet_value=DIRICHLET_VALUE,
                    x_pts=x_pts, y_pts=y_pts, maxiters=args.maxiters, verbose=False,
                    linear_solver=args.linear_solver, cg_maxiter=args.cg_maxiter,
                    cg_tol=args.cg_tol, dampfactor=args.dampfactor,
                )
                wall = time.perf_counter() - t0

                cg_per_step = cg_iters if isinstance(cg_iters, list) else [int(cg_iters)]
                cg_total    = int(sum(cg_per_step))
                final_res   = float(residuals[-1]) if residuals else float('nan')

                # SimpleNewtonSolve never raises on non-convergence — it just
                # runs to maxiters, potentially with a nan/inf residual from a
                # blown-up solution. Treat that as diverged, not converged.
                converged = np.isfinite(final_res)
                if converged:
                    n_converged += 1
                    status = 'converged'
                else:
                    n_diverged += 1
                    status = 'DIVERGED: non-finite final residual'

            except Exception as e:
                wall      = time.perf_counter() - t0
                converged = False
                nr_iters, cg_per_step, cg_total = -1, [], -1
                final_res = float('nan')
                n_diverged += 1
                status = f'FAILED: {e}'

            cg_str = ';'.join(str(c) for c in cg_per_step)
            print(f"[{i+1:>3}/{len(params)}] {split:3s}  k={k:6.2f}  A={A:6.1f}  "
                  f"n={nv:2d}  mu1={mu1:.2f}  mu2={mu2:.2f}  "
                  f"→ {status}  nr={nr_iters}  cg_total={cg_total}  "
                  f"res={final_res:.2e}  t={wall:.2f}s")

            writer.writerow({
                'idx':              i,
                'split':            split,
                'k':                round(k,   6),
                'A':                round(A,   6),
                'n':                nv,
                'mu1':              round(mu1, 6),
                'mu2':              round(mu2, 6),
                'converged':        converged,
                'nr_iters':         nr_iters,
                'final_residual':   f'{final_res:.6e}',
                'cg_iters_per_step': cg_str,
                'cg_total':         cg_total,
                'wall_s':           round(wall, 4),
            })
            f.flush()

    # ── summary ────────────────────────────────────────────────────────────
    stats = pd.read_csv(args.output)
    conv  = stats[stats['converged'] == True]
    print(f"\n{'='*60}")
    print(f"Total: {len(stats)}  converged: {n_converged}  failed: {n_diverged}")
    if len(conv):
        print(f"nr_iters  — mean={conv['nr_iters'].mean():.1f}  "
              f"min={conv['nr_iters'].min()}  max={conv['nr_iters'].max()}")
        print(f"cg_total  — mean={conv['cg_total'].mean():.1f}  "
              f"min={conv['cg_total'].min()}  max={conv['cg_total'].max()}")
        print(f"wall_s    — mean={conv['wall_s'].mean():.2f}s  total={conv['wall_s'].sum():.1f}s")

        id_conv  = conv[stats['split'] == 'id']
        ood_conv = conv[stats['split'] == 'ood']
        if len(id_conv):
            print(f"  ID  nr mean={id_conv['nr_iters'].mean():.1f}  cg mean={id_conv['cg_total'].mean():.1f}")
        if len(ood_conv):
            print(f"  OOD nr mean={ood_conv['nr_iters'].mean():.1f}  cg mean={ood_conv['cg_total'].mean():.1f}")
    print(f"{'='*60}")
    print(f"Saved → {args.output}")


if __name__ == '__main__':
    main()
