#!/usr/bin/env bash
# run_inference_all.sh
# Runs inference.py for all DoE runs on all 5 held-out test sets
# (3 ID seeds + 2 OOD seeds), parallelising across GPU 0 and GPU 1.
# Run from: 2d_possion/
# Usage:    bash run_inference_all.sh [--dry-run]

set -euo pipefail

DRY=${1:-""}
LOGDIR="logs/inference_all_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOGDIR"

# ── Held-out test sets ────────────────────────────────────────────────────────
TEST_SETS=(
    "data/test_sets/test_id_seed0.h5   test_id_seed0"
    "data/test_sets/test_id_seed1.h5   test_id_seed1"
    "data/test_sets/test_id_seed2.h5   test_id_seed2"
    "data/test_sets/test_ood_seed0.h5  test_ood_seed0"
    "data/test_sets/test_ood_seed1.h5  test_ood_seed1"
)
SAMPLE_INDICES=$(seq -s' ' 0 31)

run_infer() {
    local run_name=$1
    local cfg=$2
    local gpu=$3
    local test_h5=$4
    local out_tag=$5
    local log="$LOGDIR/${run_name}__${out_tag}.log"

    # Skip if inference_results.csv already exists for this test set
    local existing
    existing=$(find "runs/${run_name}/inference/${out_tag}" -name "inference_results.csv" 2>/dev/null | head -1 || true)
    if [[ -n "$existing" ]]; then
        echo "  [SKIP] $run_name / $out_tag — already done"
        return
    fi

    if [[ ! -f "$cfg" ]]; then
        echo "  [MISSING CONFIG] $run_name — $cfg"
        return
    fi

    local ckpt
    ckpt=$(find "runs/${run_name}" -maxdepth 3 -name "best_model.pt" 2>/dev/null | head -1 || true)
    if [[ -z "$ckpt" ]]; then
        echo "  [MISSING CKPT] $run_name — no best_model.pt"
        return
    fi

    echo "  [RUN] $run_name / $out_tag → GPU $gpu  (log: $log)"
    if [[ "$DRY" != "--dry-run" ]]; then
        python inference.py \
            --config "$cfg" \
            --gpu "$gpu" \
            --ckpt "$ckpt" \
            --test_path "$test_h5" \
            --out_tag "$out_tag" \
            --sample_indices $SAMPLE_INDICES \
            > "$log" 2>&1
    fi
}

# ── Model run list: (run_name, config) ───────────────────────────────────────
declare -a RUNS=(
# 128 dd
  "128_UNet2D_dd                  configs/unet_configs/128/config_128_dd.yaml"
  "128_UNet2D_dd_seed1            configs/unet_configs/128/config_128_dd_seed1.yaml"
  "128_UNet2D_dd_seed2            configs/unet_configs/128/config_128_dd_seed2.yaml"
# 128 M1
  "128_UNet2D_solver_M1_fixed     configs/unet_configs/128/config_128_solver_M1.yaml"
  "128_UNet2D_solver_M1_seed1     configs/unet_configs/128/config_128_solver_M1_seed1.yaml"
  "128_UNet2D_solver_M1_seed2     configs/unet_configs/128/config_128_solver_M1_seed2.yaml"
# 128 M3
  "128_UNet2D_solver_M3_fixed     configs/unet_configs/128/config_128_solver_M3.yaml"
  "128_UNet2D_solver_M3_seed1     configs/unet_configs/128/config_128_solver_M3_seed1.yaml"
  "128_UNet2D_solver_M3_seed2     configs/unet_configs/128/config_128_solver_M3_seed2.yaml"
# 128 M5
  "128_UNet2D_solver_M5_fixed     configs/unet_configs/128/config_128_solver_M5.yaml"
  "128_UNet2D_solver_M5_seed1     configs/unet_configs/128/config_128_solver_M5_seed1.yaml"
  "128_UNet2D_solver_M5_seed2     configs/unet_configs/128/config_128_solver_M5_seed2.yaml"
# 256 dd
  "256_UNet2D_dd                  configs/unet_configs/256/config_256_dd.yaml"
  "256_UNet2D_dd_seed1            configs/unet_configs/256/config_256_dd_seed1.yaml"
  "256_UNet2D_dd_seed2            configs/unet_configs/256/config_256_dd_seed2.yaml"
# 256 M1
  "256_UNet2D_solver_M1_fixed     configs/unet_configs/256/config_256_solver_M1.yaml"
  "256_UNet2D_solver_M1_seed1     configs/unet_configs/256/config_256_solver_M1_seed1.yaml"
  "256_UNet2D_solver_M1_seed2     configs/unet_configs/256/config_256_solver_M1_seed2.yaml"
# 256 M3
  "256_UNet2D_solver_M3_fixed     configs/unet_configs/256/config_256_solver_M3.yaml"
  "256_UNet2D_solver_M3_seed1     configs/unet_configs/256/config_256_solver_M3_seed1.yaml"
  "256_UNet2D_solver_M3_seed2     configs/unet_configs/256/config_256_solver_M3_seed2.yaml"
# 256 M5
  "256_UNet2D_solver_M5_fixed     configs/unet_configs/256/config_256_solver_M5.yaml"
  "256_UNet2D_solver_M5_seed1     configs/unet_configs/256/config_256_solver_M5_seed1.yaml"
  "256_UNet2D_solver_M5_seed2     configs/unet_configs/256/config_256_solver_M5_seed2.yaml"
# 512 dd
  "512_UNet2D_dd                  configs/unet_configs/512/config_512_dd.yaml"
  "512_UNet2D_dd_seed1            configs/unet_configs/512/config_512_dd_seed1.yaml"
  "512_UNet2D_dd_seed2            configs/unet_configs/512/config_512_dd_seed2.yaml"
# 512 M1
  "512_UNet2D_solver_M1_fixed     configs/unet_configs/512/config_512_solver_M1.yaml"
  "512_UNet2D_solver_M1_seed1     configs/unet_configs/512/config_512_solver_M1_seed1.yaml"
  "512_UNet2D_solver_M1_seed2     configs/unet_configs/512/config_512_solver_M1_seed2.yaml"
# 512 M3
  "512_UNet2D_solver_M3_fixed     configs/unet_configs/512/config_512_solver_M3.yaml"
  "512_UNet2D_solver_M3_seed1     configs/unet_configs/512/config_512_solver_M3_seed1.yaml"
  "512_UNet2D_solver_M3_seed2     configs/unet_configs/512/config_512_solver_M3_seed2.yaml"
# 512 M5
  "512_UNet2D_solver_M5_fixed     configs/unet_configs/512/config_512_solver_M5.yaml"
  "512_UNet2D_solver_M5_seed1     configs/unet_configs/512/config_512_solver_M5_seed1.yaml"
  "512_UNet2D_solver_M5_seed2     configs/unet_configs/512/config_512_solver_M5_seed2.yaml"
)

# Total jobs = runs × test_sets
total=$(( ${#RUNS[@]} * ${#TEST_SETS[@]} ))
echo "=== Batch inference: ${#RUNS[@]} models × ${#TEST_SETS[@]} test sets = ${total} jobs ==="
echo "    Logs → $LOGDIR"
[[ "$DRY" == "--dry-run" ]] && echo "    [DRY RUN MODE]"
echo ""

# Build flat job list: (run_name, cfg, test_h5, out_tag), alternate GPU 0/1
declare -a JOBS=()
gpu=0
for run_entry in "${RUNS[@]}"; do
    read -r rn cfg <<< "$run_entry"
    for ts_entry in "${TEST_SETS[@]}"; do
        read -r h5 tag <<< "$ts_entry"
        JOBS+=("${rn}|${cfg}|${h5}|${tag}|${gpu}")
        gpu=$(( 1 - gpu ))
    done
done

# Launch in pairs (GPU0 + GPU1), wait after each pair
i=0
while [[ $i -lt ${#JOBS[@]} ]]; do
    IFS='|' read -r rn0 cfg0 h5_0 tag0 gpu0 <<< "${JOBS[$i]}"

    if [[ $((i + 1)) -lt ${#JOBS[@]} ]]; then
        IFS='|' read -r rn1 cfg1 h5_1 tag1 gpu1 <<< "${JOBS[$((i+1))]}"
        run_infer "$rn0" "$cfg0" "$gpu0" "$h5_0" "$tag0" &
        PID0=$!
        run_infer "$rn1" "$cfg1" "$gpu1" "$h5_1" "$tag1" &
        PID1=$!
        wait $PID0; wait $PID1
        i=$((i + 2))
    else
        run_infer "$rn0" "$cfg0" "$gpu0" "$h5_0" "$tag0"
        i=$((i + 1))
    fi
done

echo ""
echo "=== Done. Results layout: runs/{model}/inference/test_{id|ood}_seed{N}/inference_results.csv ==="
echo "=== Aggregate: python aggregate_results.py --runs_dir ./runs ==="
