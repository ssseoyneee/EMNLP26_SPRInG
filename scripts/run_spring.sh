#!/bin/bash
# Full SPRInG pipeline for one task / profile size / seed.
#   bash scripts/run_spring.sh abstract_generation small 42
# Requires: HF_TOKEN exported (Gemma-3 is gated), data/<task>/final_dataset_<size>.json
# All artifacts go to runs/<task>/<size>/<model>/<hparams>/seed<seed>/ (see spring/paths.py).
# Re-running skips finished users/periods, so an interrupted job can simply be resubmitted.
set -euo pipefail

TASK=${1:-abstract_generation}
SIZE=${2:-small}
SEED=${3:-42}
MODEL=${MODEL:-google/gemma-3-4b-it}
START_USER=${START_USER:-0}
END_USER=${END_USER:-50}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

cd "$(dirname "$0")/.."
LOG_DIR=logs/${TASK}/${SIZE}/seed${SEED}/u${START_USER}-${END_USER}
mkdir -p "${LOG_DIR}"

COMMON=(--model_name "${MODEL}" --task_name "${TASK}" --profile_size "${SIZE}" --seed "${SEED}"
        --start_user "${START_USER}" --end_user "${END_USER}"
        --learning_rate 1e-4 --max_epoch 2 --batch_size 4 --r 8 --alpha 8
        --selection_mode bottom --selection_p 30 --oppu_alpha 1.0)

echo "[$(date)] Stage 1: period-0 SFT"
python -u main.py --stage sft "${COMMON[@]}" > "${LOG_DIR}/sft.out" 2> "${LOG_DIR}/sft.err"

echo "[$(date)] Stage 2: continual learning + retrieval buffer"
python -u main.py --stage cl "${COMMON[@]}" --use_retrieval_buffer --retrieval_buffer_max 50 \
  > "${LOG_DIR}/cl.out" 2> "${LOG_DIR}/cl.err"

echo "[$(date)] Stage 3: retrieval-interpolated inference"
python -u main.py --stage infer "${COMMON[@]}" --use_retrieval_buffer \
  --rb_overflow_strategy global_lowest --final_rag_mode interpolation --interp_lambda 0.5 --k 2 \
  > "${LOG_DIR}/infer.out" 2> "${LOG_DIR}/infer.err"

echo "[$(date)] Done. Run dir: $(grep -m1 'run dir' "${LOG_DIR}/infer.out" | sed 's/.*run dir: //')"
