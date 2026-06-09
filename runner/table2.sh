#!/usr/bin/env bash
# =============================================================================
# Run the Table 2 DST configs.
#
# Table 2 = JGA on MultiWOZ 2.4 in the open-source setting (Llama-3-8B-Instruct
#          and 4-bit Llama-3-70B-Instruct as the DST model), CombiSearch vs
#          RefPyDST retriever, at 1% / 5% / 100% of the training data.
#
# Pipeline shape: practical (trained-retriever) DST inference (run_dst / S4),
# identical retriever artifacts to Table 1 — only the DST LLM differs.
#
# Default: run the 12 DST configs only (retrievers/indexes assumed on disk).
# FULL=1 rebuilds from raw data (preprocess -> indexes -> CombiSearch data gen
# -> CombiSearch retriever training -> download RefPyDST retrievers).
#
#   FULL=1 bash runner/table2.sh
#   DRY_RUN=1 bash runner/table2.sh
# =============================================================================

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

STAGES="${ROOT_DIR}/runner/stages"
FULL="${FULL:-0}"

# CombiSearch data generation + retriever fine-tuning (one per training ratio).
COMBISEARCH_DATA_GEN_CONFIGS=(
  configs/data_generation/mwoz21/1p/split_v1.json
  configs/data_generation/mwoz21/5p/split_v1.json
  configs/data_generation/mwoz21/100p/split_v1.json
)
COMBISEARCH_TRAIN_CONFIGS=(
  configs/finetuning/mwoz21/combisearch/1p/split_v1.json
  configs/finetuning/mwoz21/combisearch/5p/split_v1.json
  configs/finetuning/mwoz21/combisearch/100p/split_v1.json
)

# DST inference — 8B + 70B, CombiSearch + RefPyDST, 1p/5p/100p (12 runs).
DST_CONFIGS=( configs/table_2/*/*/*/split_v*.json )

if [[ "${FULL}" == "1" ]]; then
  section "S0. Preprocess MultiWOZ"
  bash "${STAGES}/s0_preprocess.sh"

  section "S1. Build pretrained SBERT indexes"
  bash "${STAGES}/s1_build_indexes.sh"

  section "S2. CombiSearch data generation"
  bash "${STAGES}/s2_generate_data.sh" "${COMBISEARCH_DATA_GEN_CONFIGS[@]}"

  section "S3. CombiSearch retriever fine-tuning"
  bash "${STAGES}/s3_train_retriever.sh" "${COMBISEARCH_TRAIN_CONFIGS[@]}"

  section "S3-alt. Download RefPyDST retrievers from HuggingFace"
  bash "${STAGES}/download_refpydst_retrievers.sh"
fi

section "S4. DST inference — Table 2 (Llama-3 8B/70B)"
bash "${STAGES}/s4_run_dst.sh" "${DST_CONFIGS[@]}"

echo
echo "[done] Table 2 configs run. Check outputs/runs/table_2/**/running_log.json"
