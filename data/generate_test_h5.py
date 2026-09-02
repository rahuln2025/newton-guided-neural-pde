"""
Generate a held-out test .h5 dataset from a parameter CSV.

Normalization stats (fxy_stats.npz) must come from the *training* split of the
model under evaluation — each model variant has its own stats file.

Usage:
    python data/generate_test_h5.py \
        --params_csv  data/test_sets/test_params_id_seed0.csv \
        --stats_npz   data/data_256_solver/fxy_stats.npz \
        --output      data/test_sets/test_id_seed0_256s.h5 \
        --n_pts       64 \
        --maxiters    100

Solver args (cg, dampfactor) default to values used in training configs.
All Dirichlet BCs are homogeneous zero on all 4 sides (matches training).
"""

import argparse
import csv
import os
import sys

import h5py
import numpy as np
import pandas as pd

# add project root to path so solver / utils_data_pdes are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import solve_pde, sampling_points_2d_grid
from utils_data_pdes import create_unet_input_fxy


DIRICHLET_BCS   = [1, 2, 3, 4]
DIRICHLET_VALUE = 0.0


def generate(params_csv: str, stats_npz: str, output: str,
             n_pts: int, maxiters: int,
             linear_solver: str, cg_maxiter: int, cg_tol: float,
             dampfactor: float):

    params = pd.read_csv(params_csv)
    n = len(params)

    stats = np.load(stats_npz)
    mu_k, std_k = float(stats['mu_k']), float(stats['std_k'])
    mu_n, std_n = float(stats['mu_n']), float(stats['std_n'])
    mu_f, std_f = float(stats['mu_f']), float(stats['std_f'])
    print(f"Stats loaded from {stats_npz}")
    print(f"  mu_k={mu_k:.4f} std_k={std_k:.4f}  mu_n={mu_n:.4f} std_n={std_n:.4f}")
    print(f"  mu_f={mu_f:.4f} std_f={std_f:.4f}")

    x_pts, y_pts, _, _, _, _ = sampling_points_2d_grid(n_pts=n_pts)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    csv_path = output.replace('.h5', '_solver_stats.csv')

    with h5py.File(output, 'w') as f, \
         open(csv_path, 'w', newline='') as csv_file:

        f.create_dataset('input_tensor',    (n, 4, n_pts, n_pts), dtype=np.float32)
        f.create_dataset('solution_tensor', (n, 1, n_pts, n_pts), dtype=np.float64)
        f.create_dataset('k',              (n,), dtype=np.float64)
        f.create_dataset('A',              (n,), dtype=np.float64)
        f.create_dataset('n',              (n,), dtype=np.int32)
        f.create_dataset('mu1',            (n,), dtype=np.float64)
        f.create_dataset('mu2',            (n,), dtype=np.float64)
        f.create_dataset('boundary_conditions', (n,),
                         dtype=h5py.special_dtype(vlen=np.int32))
        f.create_dataset('dirichlet_value', (n,), dtype=np.float64)
        f.create_dataset('sample_index',    (n,), dtype=np.int32)

        writer = csv.DictWriter(csv_file,
                                fieldnames=['sample_index', 'k', 'n', 'A',
                                            'mu1', 'mu2', 'solver_iters', 'cg_iters'])
        writer.writeheader()

        for i, row in params.iterrows():
            k   = float(row['k'])
            A   = float(row['A'])
            nv  = int(row['n'])
            mu1 = float(row['mu1'])
            mu2 = float(row['mu2'])

            print(f"[{i+1}/{n}]  k={k:.3f}  A={A:.3f}  n={nv}  mu1={mu1:.3f}  mu2={mu2:.3f}")

            input_tensor = create_unet_input_fxy(
                k=k, n=nv, A=A, mu1=mu1, mu2=mu2,
                boundary_conditions=DIRICHLET_BCS,
                bc_value=DIRICHLET_VALUE,
                grid_size=n_pts,
                mu_k=mu_k, std_k=std_k,
                mu_n=mu_n, std_n=std_n,
                mu_f=mu_f, std_f=std_f,
            )

            u_sol, iters, _, cg_iters, _, _ = solve_pde(
                u_init=None,
                k=k, n=nv, A=A, mu1=mu1, mu2=mu2,
                dirichlet_bcs=DIRICHLET_BCS,
                dirichlet_value=DIRICHLET_VALUE,
                x_pts=x_pts, y_pts=y_pts,
                maxiters=maxiters,
                verbose=False,
                linear_solver=linear_solver,
                cg_maxiter=cg_maxiter,
                cg_tol=cg_tol,
                dampfactor=dampfactor,
            )
            print(f"         solver iters={iters}  cg_iters={cg_iters}")

            f['input_tensor'][i]        = input_tensor
            f['solution_tensor'][i]     = u_sol
            f['k'][i]                   = k
            f['A'][i]                   = A
            f['n'][i]                   = nv
            f['mu1'][i]                 = mu1
            f['mu2'][i]                 = mu2
            f['boundary_conditions'][i] = DIRICHLET_BCS
            f['dirichlet_value'][i]     = DIRICHLET_VALUE
            f['sample_index'][i]        = i

            cg_total = int(sum(cg_iters)) if isinstance(cg_iters, list) else int(cg_iters)
            writer.writerow({'sample_index': i, 'k': round(k, 4), 'n': nv,
                             'A': round(A, 4), 'mu1': round(mu1, 4), 'mu2': round(mu2, 4),
                             'solver_iters': iters, 'cg_iters': cg_total})
            csv_file.flush()

    print(f"\nSaved {n} samples → {output}")
    print(f"Solver stats   → {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_csv',    required=True)
    parser.add_argument('--stats_npz',     required=True,
                        help='fxy_stats.npz from the training split of the target model')
    parser.add_argument('--output',        required=True)
    parser.add_argument('--n_pts',         type=int,   default=64)
    parser.add_argument('--maxiters',      type=int,   default=100)
    parser.add_argument('--linear_solver', default='cg')
    parser.add_argument('--cg_maxiter',    type=int,   default=2000)
    parser.add_argument('--cg_tol',        type=float, default=1e-12)
    parser.add_argument('--dampfactor',    type=float, default=1.0)
    args = parser.parse_args()

    generate(
        params_csv=args.params_csv,
        stats_npz=args.stats_npz,
        output=args.output,
        n_pts=args.n_pts,
        maxiters=args.maxiters,
        linear_solver=args.linear_solver,
        cg_maxiter=args.cg_maxiter,
        cg_tol=args.cg_tol,
        dampfactor=args.dampfactor,
    )


if __name__ == '__main__':
    main()
