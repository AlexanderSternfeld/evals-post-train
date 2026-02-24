#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --cpus-per-task=200
#SBATCH --gres=gpu:4
#SBATCH --job-name=eval-wmt-translation
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --time=05:59:00

# -----------------------------------------------------------------------
# Test script for WMT translation benchmark (CometKiwi22 metric).
#
# Installs from the swiss-ai/lm-evaluation-harness-wmt GitHub fork
# plus subset2evaluate (data) and unbabel-comet (metric).
#
# Usage:
#   sbatch scripts/evaluate_wmt_translation.sbatch <model> <name> [limit]
#
# Examples:
#   # Quick sanity check (10 items)
#   sbatch scripts/evaluate_wmt_translation.sbatch swiss-ai/Apertus-8B-2509 apertus-8b 10
#
#   # Full run (all items, ~332 per language pair)
#   sbatch scripts/evaluate_wmt_translation.sbatch swiss-ai/Apertus-8B-2509 apertus-8b
# -----------------------------------------------------------------------

set -e
echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"

# --- Arguments ---
if (( $# < 2 )); then
    echo "Usage: sbatch evaluate_wmt_translation.sbatch <model> <name> [limit]"
    exit 1
fi
MODEL=$1
NAME=$2
HARNESS_LIMIT=${3:-""}

# --- Configuration ---
TOKENIZER=${TOKENIZER:-$MODEL}
BS=${BS:-"auto:20"}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}

# Tasks: run a small subset of WMT25 for testing, or override with TASKS env var
TASKS=${TASKS:-"wmt25_en-zh_CN"}

# W&B configuration
WANDB_ENTITY=${WANDB_ENTITY:-matinansaripour}
WANDB_PROJECT=${WANDB_PROJECT:-wmt-translation-test}
TABLE_METRICS=${TABLE_METRICS:-"wmt25_en-zh_CN/cometkiwi22"}

export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < ./scripts/hf_token.txt 2>/dev/null || echo '')}"
export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d '\r\n' < ./scripts/wandb_api_key.txt 2>/dev/null || echo '')}"
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/$USER/huggingface}

# lm-evaluation-harness fork with wmt_translation task
LM_EVAL_HARNESS_REPO="https://github.com/swiss-ai/lm-evaluation-harness-wmt.git"

# --- Logging ---
LOGS_ROOT=${LOGS_ROOT:-/iopsstor/scratch/cscs/$USER/eval-logs}
RUN_ROOT=$LOGS_ROOT/$WANDB_ENTITY/$WANDB_PROJECT/$NAME
HARNESS_DIR=$RUN_ROOT/harness
export WANDB_DIR=$RUN_ROOT
mkdir -p $HARNESS_DIR

HARNESS_EVAL_DIR=$HARNESS_DIR/eval_$(date +%Y%m%d_%H%M%S)_$SLURM_JOBID

# --- Print config ---
echo "Configuration:"
echo "  MODEL=$MODEL"
echo "  NAME=$NAME"
echo "  TOKENIZER=$TOKENIZER"
echo "  TASKS=$TASKS"
echo "  HARNESS_LIMIT=$HARNESS_LIMIT"
echo "  LM_EVAL_HARNESS_REPO=$LM_EVAL_HARNESS_REPO"
echo "  APPLY_CHAT_TEMPLATE=$APPLY_CHAT_TEMPLATE"
echo "  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "  WANDB=$WANDB_ENTITY/$WANDB_PROJECT"
echo "  OUTPUT=$HARNESS_EVAL_DIR"

# --- vLLM environment ---
GPUS_PER_NODE=4
MP=$GPUS_PER_NODE
DP=$(( GPUS_PER_NODE / MP ))
VLLM_MEMORY_FRACTION=0.75

export MASTER_ADDR=$(hostname)
export MASTER_PORT=25678
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WORLD_SIZE=$GPUS_PER_NODE
export OMP_NUM_THREADS=$(( 8 * $GPUS_PER_NODE ))
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_COMPILE_CACHE=1
export HF_ALLOW_CODE_EVAL=1

# --- Model args ---
COMMON_MODEL_ARGS="dtype=bfloat16,pretrained=$MODEL,tokenizer=$TOKENIZER,data_parallel_size=$DP,tensor_parallel_size=$MP,gpu_memory_utilization=$VLLM_MEMORY_FRACTION,enable_thinking=False"

# --- Eval args ---
COMMON_EVAL_ARGS=(
    --trust_remote_code
    --batch_size $BS
    --tasks $TASKS
    --output_path $HARNESS_EVAL_DIR
    --max_batch_size 32
    --log_samples
    --write_out
    --confirm_run_unsafe_code
    --cache_requests true
    --gen_kwargs max_gen_toks=$MAX_NEW_TOKENS
)

if [[ -n "${HARNESS_LIMIT}" ]]; then
    COMMON_EVAL_ARGS+=(--limit $HARNESS_LIMIT)
fi

if [[ $APPLY_CHAT_TEMPLATE = true ]]; then
    COMMON_EVAL_ARGS+=(--apply_chat_template)
    COMMON_EVAL_ARGS+=(--fewshot_as_multiturn)
fi

CMD="lm_eval --model vllm --model_args=$COMMON_MODEL_ARGS ${COMMON_EVAL_ARGS[@]}"

# --- Install command ---
INSTALL_CMD="pip install --no-cache-dir --upgrade \"git+${LM_EVAL_HARNESS_REPO}\" && \
pip install --no-cache-dir \
    subset2evaluate \
    unbabel-comet && \
pip install --no-cache-dir --upgrade \
    \"huggingface-hub>=1.3.0,<2.0\" \
    \"transformers==5.1.0\" \
    \"accelerate>=1.1.0\" \
    \"antlr4-python3-runtime==4.11\" \
    tokenizers \
    hf-transfer \
    \"peft>=0.2.0\""

echo "Install command: $INSTALL_CMD"
echo "Eval command: $CMD"

# --- Run ---
START_TIME=$(date +%s)

srun -n 1 --mpi=pmix -ul --environment=./containers/env_vllm.toml bash -lc "
    set -euo pipefail
    unset SSL_CERT_FILE

    $INSTALL_CMD

    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}

    $CMD
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Time taken: ${DURATION}s ($(($DURATION / 60))m $(($DURATION % 60))s)"

# --- Upload to W&B ---
echo "Uploading results to wandb"
WANDB_CMD="cd $PWD && python -m scripts.alignment.update_wandb_alignment --entity $WANDB_ENTITY --project $WANDB_PROJECT --logs_root $HARNESS_EVAL_DIR --name $NAME --main_metrics $TABLE_METRICS --eval_duration $DURATION"
echo "W&B command: $WANDB_CMD"
srun -ul --mpi=pmix --environment=./containers/env.toml bash -c " \
    $WANDB_CMD
"

echo "Evaluation finished"
echo "END TIME: $(date)"
