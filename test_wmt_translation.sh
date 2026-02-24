#!/bin/bash
# Test WMT translation benchmark on Apertus 8B.
#
# Usage:
#   cd /iopsstor/scratch/cscs/ansaripo/dev_evals/evals-post-train
#
#   # Quick sanity check (10 items, ~5-10 min)
#   bash test_wmt_translation.sh
#
#   # Full single language pair (332 items)
#   bash test_wmt_translation.sh full
#
#   # Full WMT25 suite (16 language pairs)
#   bash test_wmt_translation.sh wmt25

set -euo pipefail

MODE=${1:-all}
MODEL="swiss-ai/Apertus-8B-2509"
NAME="apertus-8b-2509"

export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
export HF_TOKEN="<YOUR_HF_TOKEN>"
export HF_HOME="<YOUR_HF_HOME>"
export WANDB_PROJECT="<YOUR_WANDB_PROJECT>"
export APPLY_CHAT_TEMPLATE=false

case "$MODE" in
    quick)
        # Single language pair, 10 items — fast sanity check
        export TASKS="wmt25_en-zh_CN"
        export TABLE_METRICS="wmt25_en-zh_CN/cometkiwi22"
        LIMIT=10
        ;;
    wmt25)
        # All WMT25 language pairs
        export TASKS="wmt25_translation"
        export TABLE_METRICS="wmt25_translation/cometkiwi22"
        LIMIT=""
        ;;
    all)
        # Full WMT translation suite (wmt24 + wmt24pp + wmt25, 82 language pairs)
        export TASKS="wmt_translation"
        export TABLE_METRICS="wmt_translation/cometkiwi22"
        LIMIT=""
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash test_wmt_translation.sh [quick|wmt25|all]"
        exit 1
        ;;
esac

echo "======================================"
echo "WMT Translation Test"
echo "  Mode:  $MODE"
echo "  Model: $MODEL"
echo "  Tasks: $TASKS"
echo "  Limit: ${LIMIT:-none}"
echo "======================================"

sbatch --job-name "eval-wmt-$MODE" \
    scripts/evaluate_wmt_translation.sbatch \
    "$MODEL" "$NAME" "$LIMIT"
