"""
Generate a pool of 300 held-out test parameters, disjoint from sweep_results.csv.

  200 in-distribution (ID)  — same ranges as training config
  100 out-of-distribution (OOD) — strictly outside training bounds on ≥1 axis

ID ranges  (training config):
    k   ∈ [0.2,  20.0]
    A   ∈ [1.0,  80.0]
    mu1 ∈ [1.0,   4.0]
    mu2 ∈ [1.0,   4.0]
    n   ∈ {3, 5, 7, 9, 11, 13}

OOD ranges (at least one parameter outside training bounds):
    k   ∈ [20.0, 40.0]   OR
    A   ∈ [80.0, 120.0]  OR
    mu  ∈ [4.0,   6.0]   OR
    n   ∈ {15, 17, 19}
  OOD samples are drawn uniformly from the extended space and then
  filtered to keep only those with ≥1 parameter outside training bounds.

Outputs:
    data/test_sets/test_params_pool.csv          (300 samples, split='id'/'ood' column)
    data/test_sets/test_params_id_seed{0,1,2}.csv   (32 samples each, non-overlapping, ID only)
    data/test_sets/test_params_ood_seed{0,1,2}.csv  (32 samples each, non-overlapping, OOD only)

Usage:
    python data/generate_test_params.py \
        --sweep_csv        data/sweep_results.csv \
        --out_dir          data/test_sets \
        --solver_stats_csv data/test_sets/pool_solver_stats.csv \
        --n_id             200 \
        --n_ood            100 \
        --n_test           32 \
        --n_splits         3 \
        --rng_seed         9999

Note:
    By default, already-existing test_params_{id,ood}_seed{s}.csv files in
    --out_dir are left untouched (not overwritten) so that previously
    generated/used test splits stay reproducible. Pass --force to regenerate
    them (e.g. after intentionally changing sampling/filtering strategy).
"""

import argparse
import os
import numpy as np
import pandas as pd

# ── Training (in-distribution) bounds ────────────────────────────────────────
ID_K_RANGE   = (0.2,  20.0)
ID_A_RANGE   = (1.0,  80.0)
ID_MU_RANGE  = (1.0,   4.0)
ID_N_CHOICES = [3, 5, 7, 9, 11, 13]

# ── Extended bounds for OOD sampling ─────────────────────────────────────────
OOD_K_RANGE   = (0.2,  40.0)
OOD_A_RANGE   = (1.0, 120.0)
OOD_MU_RANGE  = (1.0,   6.0)
OOD_N_CHOICES = [3, 5, 7, 9, 11, 13, 15, 17, 19]


def _is_ood(row: pd.Series) -> bool:
    return (
        row['k']   > ID_K_RANGE[1]  or
        row['A']   > ID_A_RANGE[1]  or
        row['mu1'] > ID_MU_RANGE[1] or
        row['mu2'] > ID_MU_RANGE[1] or
        row['n']   not in ID_N_CHOICES
    )


def sample_id(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame({
        'k':   rng.uniform(*ID_K_RANGE,  size=n),
        'A':   rng.uniform(*ID_A_RANGE,  size=n),
        'mu1': rng.uniform(*ID_MU_RANGE, size=n),
        'mu2': rng.uniform(*ID_MU_RANGE, size=n),
        'n':   rng.choice(ID_N_CHOICES,  size=n).astype(int),
    })
    df['split'] = 'id'
    return df


def sample_ood(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample from extended space, keep only genuinely OOD rows."""
    collected = []
    batch = max(n * 5, 500)
    while sum(len(c) for c in collected) < n:
        cand = pd.DataFrame({
            'k':   rng.uniform(*OOD_K_RANGE,  size=batch),
            'A':   rng.uniform(*OOD_A_RANGE,  size=batch),
            'mu1': rng.uniform(*OOD_MU_RANGE, size=batch),
            'mu2': rng.uniform(*OOD_MU_RANGE, size=batch),
            'n':   rng.choice(OOD_N_CHOICES,  size=batch).astype(int),
        })
        cand = cand[cand.apply(_is_ood, axis=1)]
        collected.append(cand)
    df = pd.concat(collected, ignore_index=True).iloc[:n]
    df['split'] = 'ood'
    return df


def check_overlap(pool: pd.DataFrame, existing: pd.DataFrame, tol: float = 1e-4) -> int:
    cont_cols = ['k', 'A', 'mu1', 'mu2']
    pool_arr  = pool[cont_cols].values
    exist_arr = existing[cont_cols].values
    count = 0
    for row in pool_arr:
        if np.sqrt(((exist_arr - row) ** 2).sum(axis=1)).min() < tol:
            count += 1
    return count


def filter_diverged(df: pd.DataFrame, solver_stats_csv: str) -> pd.DataFrame:
    """Drop rows from df that appear as diverged (converged == False) in solver_stats_csv.

    solve_pool.py writes k/A/mu1/mu2 rounded to 6 decimals, so we round the
    same columns here before merging — an exact merge against the raw
    high-precision floats would never match and silently drop nothing.
    """
    stats = pd.read_csv(solver_stats_csv)
    bad   = stats[stats['converged'] == False][['k', 'A', 'n', 'mu1', 'mu2']].copy()
    if bad.empty:
        return df

    cont_cols = ['k', 'A', 'mu1', 'mu2']
    bad[cont_cols] = bad[cont_cols].round(6)

    tmp = df.copy()
    tmp[cont_cols] = tmp[cont_cols].round(6)

    merged = tmp.merge(bad, on=['k', 'A', 'n', 'mu1', 'mu2'], how='left', indicator=True)
    mask   = merged['_merge'] == 'left_only'
    n_dropped = (~mask).sum()
    print(f"  Filtered {n_dropped} diverged rows from pool using {solver_stats_csv}")
    return df[mask.values].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_csv',        default='data/sweep_results.csv')
    parser.add_argument('--out_dir',          default='data/test_sets')
    parser.add_argument('--solver_stats_csv', default=None,
                        help='pool_solver_stats.csv — diverged rows are excluded from splits. '
                             'Defaults to <out_dir>/pool_solver_stats.csv if that file exists.')
    parser.add_argument('--n_id',      type=int, default=200)
    parser.add_argument('--n_ood',        type=int, default=100)
    parser.add_argument('--n_test',       type=int, default=32)
    parser.add_argument('--n_splits_id',  type=int, default=3,
                        help='number of non-overlapping ID test splits to write')
    parser.add_argument('--n_splits_ood', type=int, default=2,
                        help='number of non-overlapping OOD test splits to write')
    parser.add_argument('--rng_seed',     type=int, default=9999)
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing test_params_{id,ood}_seed{s}.csv files in '
                             '--out_dir. By default these are left untouched if already present, '
                             'to keep previously generated/used test splits reproducible.')
    args = parser.parse_args()

    # Auto-detect the solver stats file co-located with the output dir if the
    # caller didn't explicitly pass one (previously this silently disabled
    # filtering entirely, since the default was None).
    if args.solver_stats_csv is None:
        candidate = os.path.join(args.out_dir, 'pool_solver_stats.csv')
        if os.path.exists(candidate):
            args.solver_stats_csv = candidate
            print(f"Auto-detected solver stats: {candidate}")
        else:
            print(f"WARNING: no --solver_stats_csv given and {candidate} not found — "
                  f"diverged samples will NOT be filtered.")

    rng = np.random.default_rng(args.rng_seed)
    id_df  = sample_id(args.n_id,  rng)
    ood_df = sample_ood(args.n_ood, rng)
    pool   = pd.concat([id_df, ood_df], ignore_index=True)

    existing  = pd.read_csv(args.sweep_csv)
    n_overlap = check_overlap(pool, existing)
    print(f"Pool size      : {len(pool)}  (ID={len(id_df)}, OOD={len(ood_df)})")
    print(f"Overlap check  : {n_overlap} samples within tol=1e-4 of sweep_results.csv")
    if n_overlap > 0:
        print("WARNING: near-duplicate samples detected — consider changing --rng_seed")

    # filter diverged samples before splitting
    if args.solver_stats_csv:
        id_df  = filter_diverged(id_df,  args.solver_stats_csv)
        ood_df = filter_diverged(ood_df, args.solver_stats_csv)

    assert len(id_df)  >= args.n_test * args.n_splits_id, \
        f"ID pool too small after filtering: {len(id_df)} < {args.n_test} × {args.n_splits_id}"
    assert len(ood_df) >= args.n_test * args.n_splits_ood, \
        f"OOD pool too small after filtering: {len(ood_df)} < {args.n_test} × {args.n_splits_ood}"

    os.makedirs(args.out_dir, exist_ok=True)
    pool_path = os.path.join(args.out_dir, 'test_params_pool.csv')
    pool.to_csv(pool_path, index=False)
    print(f"Pool saved     : {pool_path}")

    for split_name, split_df, n_splits in [('id', id_df, args.n_splits_id), ('ood', ood_df, args.n_splits_ood)]:
        idx = rng.permutation(len(split_df))
        for s in range(n_splits):
            out_path = os.path.join(args.out_dir, f'test_params_{split_name}_seed{s}.csv')
            if os.path.exists(out_path) and not args.force:
                print(f"  {split_name} seed{s}: {out_path} already exists — skipped (use --force to regenerate)")
                continue
            rows = split_df.iloc[idx[s * args.n_test : (s + 1) * args.n_test]].reset_index(drop=True)
            rows.to_csv(out_path, index=False)
            n_dist = rows['n'].value_counts().sort_index().to_dict()
            print(f"  {split_name} seed{s}: {len(rows)} samples → {out_path}  | n dist: {n_dist}")

    print("\nDone.")


if __name__ == '__main__':
    main()
