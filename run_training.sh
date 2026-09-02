#!/usr/bin/env bash
# run_training_all_seeds.sh
# Launch training for a given variant and dataset size across GPU 0 and GPU 1.
# Runs seed0 and seed1 sequentially on GPU0, seed2 on GPU1, in parallel.
# Usage: bash run_training_all_seeds.sh <variant> <size>
#   variant: dd | M1 | M3 | M5
#   size:    128 | 256 | 512

set -euo pipefail

VARIANT=${1:-""}
SIZE=${2:-""}

if [[ -z "$VARIANT" || -z "$SIZE" ]]; then
    echo "Usage: bash run_training_all_seeds.sh <variant> <size>"
    echo "  variant: dd | M1 | M3 | M5"
    echo "  size:    128 | 256 | 512"
    exit 1
fi

if [[ ! "$VARIANT" =~ ^(dd|M1|M3|M5)$ ]]; then
    echo "ERROR: variant must be one of: dd M1 M3 M5"; exit 1
fi
if [[ ! "$SIZE" =~ ^(128|256|512)$ ]]; then
    echo "ERROR: size must be one of: 128 256 512"; exit 1
fi

if [[ "$VARIANT" == "dd" ]]; then
    TRAIN_SCRIPT="train_datadriven.py"
else
    TRAIN_SCRIPT="train.py"
fi

CFGBASE="configs/unet_configs/${SIZE}"

if [[ "$VARIANT" == "dd" ]]; then
    CFGS=(
        "${CFGBASE}/config_${SIZE}_dd.yaml"
        "${CFGBASE}/config_${SIZE}_dd_seed1.yaml"
        "${CFGBASE}/config_${SIZE}_dd_seed2.yaml"
    )
else
    CFGS=(
        "${CFGBASE}/config_${SIZE}_solver_${VARIANT}.yaml"
        "${CFGBASE}/config_${SIZE}_solver_${VARIANT}_seed1.yaml"
        "${CFGBASE}/config_${SIZE}_solver_${VARIANT}_seed2.yaml"
    )
fi

SEEDS=(0 1 2)
GPUS=(0 0 1)   # seed0→GPU0 (seq), seed1→GPU0 (seq), seed2→GPU1 (parallel)

echo "=== Training: variant=${VARIANT}  size=${SIZE}  script=${TRAIN_SCRIPT} ==="
for i in 0 1 2; do
    cfg="${CFGS[$i]}"
    if [[ ! -f "$cfg" ]]; then
        echo "ERROR: config not found: $cfg"; exit 1
    fi
    echo "  seed${SEEDS[$i]} → GPU ${GPUS[$i]} → $cfg"
done
echo ""

LOGDIR="logs/training_${SIZE}_${VARIANT}_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOGDIR"

# ── GPU0 branch: seed0 then seed1, sequential ────────────────────────────────
(
    echo "--- Launching seed0 on GPU ${GPUS[0]} ---"
    python "$TRAIN_SCRIPT" --config "${CFGS[0]}" --gpu "${GPUS[0]}" \
        > "${LOGDIR}/seed0.log" 2>&1
    echo "  seed0 done."

    echo "--- Launching seed1 on GPU ${GPUS[1]} ---"
    python "$TRAIN_SCRIPT" --config "${CFGS[1]}" --gpu "${GPUS[1]}" \
        > "${LOGDIR}/seed1.log" 2>&1
    echo "  seed1 done."
) &
PID_GPU0=$!

# ── GPU1 branch: seed2 ────────────────────────────────────────────────────────
echo "--- Launching seed2 on GPU ${GPUS[2]} ---"
python "$TRAIN_SCRIPT" --config "${CFGS[2]}" --gpu "${GPUS[2]}" \
    > "${LOGDIR}/seed2.log" 2>&1 &
PID_GPU1=$!

echo ""
echo "--- Waiting for GPU0 branch (PID $PID_GPU0) and GPU1 branch (PID $PID_GPU1)... ---"
wait $PID_GPU0 && echo "  GPU0 branch (seed0+seed1) done." || echo "  ERROR: GPU0 branch failed — check ${LOGDIR}/seed0.log or seed1.log"
wait $PID_GPU1 && echo "  GPU1 branch (seed2) done." || echo "  ERROR: seed2 failed — check ${LOGDIR}/seed2.log"

echo ""
echo "=== seed0 + seed1 + seed2 complete. Logs: ${LOGDIR}/ ==="