#!/usr/bin/env bash
# Fine-tune a retriever for each given config (one torchrun DDP job per config).
#
# Usage:
#   bash runner/stages/s3_train_retriever.sh <config.json> [<config.json> ...]
#
# NUM_GPUS sets processes per job (default 4); set CUDA_VISIBLE_DEVICES to pick GPUs.
# fp16 is enabled in-code (use_amp); rank-0 gates handle eval/saving.

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

NUM_GPUS="${NUM_GPUS:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"

for cfg in "$@"; do
  echo "[train_retriever] ${cfg}"
  run "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
    "${ROOT_DIR}/src/combisearch/cli/train_retriever.py" "${cfg}"
done
