#!/bin/bash
# Test WMT translation benchmark on Llama 3.1 8B base (baseline).
#
# Usage:
#   cd /iopsstor/scratch/cscs/ansaripo/dev_evals/evals-post-train
#
#   # Full WMT suite (default)
#   bash test_wmt_translation_llama.sh
#
#   # Quick sanity check (10 items)
#   bash test_wmt_translation_llama.sh quick

set -euo pipefail

MODE=${1:-all}
MODEL="meta-llama/Llama-3.1-8B"
NAME="llama-3.1-8b"

export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
export HF_TOKEN="<YOUR_HF_TOKEN>"
export HF_HOME="<YOUR_HF_HOME>"
export WANDB_PROJECT="<YOUR_WANDB_PROJECT>"
export APPLY_CHAT_TEMPLATE=false

case "$MODE" in
    quick)
        export TASKS="wmt25_en-zh_CN"
        export TABLE_METRICS="wmt25_en-zh_CN/cometkiwi22"
        LIMIT=10
        ;;
    wmt25)
        export TASKS="wmt25_translation"
        export TABLE_METRICS="wmt25_translation/cometkiwi22"
        LIMIT=""
        ;;
    all)
        export TASKS="wmt_translation"
        export TABLE_METRICS="wmt_translation/cometkiwi22"
        LIMIT=""
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash test_wmt_translation_llama.sh [quick|wmt25|all]"
        exit 1
        ;;
esac

echo "======================================"
echo "WMT Translation Test — Llama 3.1 8B base (baseline)"
echo "  Mode:  $MODE"
echo "  Model: $MODEL"
echo "  Tasks: $TASKS"
echo "  Limit: ${LIMIT:-none}"
echo "======================================"

sbatch --job-name "eval-wmt-llama-$MODE" \
    scripts/evaluate_wmt_translation.sbatch \
    "$MODEL" "$NAME" "$LIMIT"
