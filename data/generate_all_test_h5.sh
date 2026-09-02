#!/usr/bin/env bash
# Generate all 6 held-out test .h5 files (id_seed{0,1,2} + ood_seed{0,1,2})
# for a single model variant's normalization stats.
#
# Usage:
#   bash data/generate_all_test_h5.sh \
#       data/data_256_solver/fxy_stats.npz \
#       data/test_sets \
#       64
#
# Args:
#   $1  stats_npz   — fxy_stats.npz from the model's training split
#   $2  test_dir    — directory containing test_params_*.csv and .h5 outputs
#   $3  n_pts       — grid resolution (default 64)

set -e

STATS_NPZ=${1:?stats_npz required}
TEST_DIR=${2:-data/test_sets}
N_PTS=${3:-64}

for SEED in 0 1 2; do
    CSV="${TEST_DIR}/test_params_id_seed${SEED}.csv"
    OUT="${TEST_DIR}/test_id_seed${SEED}.h5"
    echo "=== Generating ${OUT} ==="
    python data/generate_test_h5.py \
        --params_csv  "${CSV}" \
        --stats_npz   "${STATS_NPZ}" \
        --output      "${OUT}" \
        --n_pts       "${N_PTS}"
done

for SEED in 0 1; do
    CSV="${TEST_DIR}/test_params_ood_seed${SEED}.csv"
    OUT="${TEST_DIR}/test_ood_seed${SEED}.h5"
    echo "=== Generating ${OUT} ==="
    python data/generate_test_h5.py \
        --params_csv  "${CSV}" \
        --stats_npz   "${STATS_NPZ}" \
        --output      "${OUT}" \
        --n_pts       "${N_PTS}"
done

echo "All test datasets generated in ${TEST_DIR}/"
