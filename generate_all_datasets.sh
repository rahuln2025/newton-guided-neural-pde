#!/usr/bin/env bash
# generate_all_datasets.sh
# Generates all seed1/seed2 datasets for dd and solver variants.
# Run from: 2d_possion/
# Usage:    bash generate_all_datasets.sh [--dry-run]

set -euo pipefail

PYTHON="python utils_data_pdes.py"
CFGBASE="configs/unet_configs"
DRY=${1:-""}

run() {
    local cfg=$1
    echo ""
    echo ">>> $cfg"
    if [[ "$DRY" == "--dry-run" ]]; then
        echo "    [DRY RUN] python utils_data_pdes.py --config $cfg"
    else
        $PYTHON --config "$cfg"
    fi
}

# ── 128 samples ──────────────────────────────────────────────────────────────
echo "=== 128: datadriven seed0 ==="
[[ ! -f "data/data_128_datadriven/train_128_64_dd.h5" ]] \
    && run "$CFGBASE/128/config_128_dd.yaml" \
    || echo "    Already exists, skipping."

echo "=== 128: solver seed0 ==="
[[ ! -f "data/data_128_solver/train_128_64_s.h5" ]] \
    && run "$CFGBASE/128/config_128_solver_M3.yaml" \
    || echo "    Already exists, skipping."

echo "=== 128: datadriven seed1 ==="
[[ ! -f "data/data_128_datadriven_seed1/train_128_64_dd.h5" ]] \
    && run "$CFGBASE/128/config_128_dd_seed1.yaml" \
    || echo "    Already exists, skipping."

echo "=== 128: datadriven seed2 ==="
[[ ! -f "data/data_128_datadriven_seed2/train_128_64_dd.h5" ]] \
    && run "$CFGBASE/128/config_128_dd_seed2.yaml" \
    || echo "    Already exists, skipping."

echo "=== 128: solver seed1 ==="
[[ ! -f "data/data_128_solver_seed1/train_128_64_s.h5" ]] \
    && run "$CFGBASE/128/config_128_solver_M3_seed1.yaml" \
    || echo "    Already exists, skipping."

echo "=== 128: solver seed2 ==="
[[ ! -f "data/data_128_solver_seed2/train_128_64_s.h5" ]] \
    && run "$CFGBASE/128/config_128_solver_M3_seed2.yaml" \
    || echo "    Already exists, skipping."

# ── 256 samples ──────────────────────────────────────────────────────────────
echo "=== 256: datadriven seed0 ==="
[[ ! -f "data/data_256_datadriven/train_256_64_dd.h5" ]] \
    && run "$CFGBASE/256/config_256_dd.yaml" \
    || echo "    Already exists, skipping."

echo "=== 256: solver seed0 ==="
[[ ! -f "data/data_256_solver/train_256_64_s.h5" ]] \
    && run "$CFGBASE/256/config_256_solver_M3.yaml" \
    || echo "    Already exists, skipping."

echo "=== 256: datadriven seed1 ==="
[[ ! -f "data/data_256_datadriven_seed1/train_256_64_dd.h5" ]] \
    && run "$CFGBASE/256/config_256_dd_seed1.yaml" \
    || echo "    Already exists, skipping."

echo "=== 256: datadriven seed2 ==="
[[ ! -f "data/data_256_datadriven_seed2/train_256_64_dd.h5" ]] \
    && run "$CFGBASE/256/config_256_dd_seed2.yaml" \
    || echo "    Already exists, skipping."

echo "=== 256: solver seed1 ==="
[[ ! -f "data/data_256_solver_seed1/train_256_64_s.h5" ]] \
    && run "$CFGBASE/256/config_256_solver_M3_seed1.yaml" \
    || echo "    Already exists, skipping."

echo "=== 256: solver seed2 ==="
[[ ! -f "data/data_256_solver_seed2/train_256_64_s.h5" ]] \
    && run "$CFGBASE/256/config_256_solver_M3_seed2.yaml" \
    || echo "    Already exists, skipping."

# ── 512 samples ──────────────────────────────────────────────────────────────
echo "=== 512: datadriven seed0 ==="
[[ ! -f "data/data_512_datadriven/train_512_64_dd.h5" ]] \
    && run "$CFGBASE/512/config_512_dd.yaml" \
    || echo "    Already exists, skipping."

echo "=== 512: solver seed0 ==="
[[ ! -f "data/data_512_solver/train_512_64_s.h5" ]] \
    && run "$CFGBASE/512/config_512_solver_M3.yaml" \
    || echo "    Already exists, skipping."

echo "=== 512: datadriven seed1 ==="
[[ ! -f "data/data_512_datadriven_seed1/train_512_64_dd.h5" ]] \
    && run "$CFGBASE/512/config_512_dd_seed1.yaml" \
    || echo "    Already exists, skipping."

echo "=== 512: datadriven seed2 ==="
[[ ! -f "data/data_512_datadriven_seed2/train_512_64_dd.h5" ]] \
    && run "$CFGBASE/512/config_512_dd_seed2.yaml" \
    || echo "    Already exists, skipping."

echo "=== 512: solver seed1 ==="
[[ ! -f "data/data_512_solver_seed1/train_512_64_s.h5" ]] \
    && run "$CFGBASE/512/config_512_solver_M3_seed1.yaml" \
    || echo "    Already exists, skipping."

echo "=== 512: solver seed2 ==="
[[ ! -f "data/data_512_solver_seed2/train_512_64_s.h5" ]] \
    && run "$CFGBASE/512/config_512_solver_M3_seed2.yaml" \
    || echo "    Already exists, skipping."

echo ""
echo "=== Done ==="
echo "Per-model train/val/test datasets generated."
echo "NOTE: common ID/OOD eval test sets are generated separately —"
echo "      see data/generate_test_params.py + data/generate_all_test_h5.sh"

