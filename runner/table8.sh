#!/usr/bin/env bash
# =============================================================================
# Run the Table 8 DST configs (individual vs combinatorial, practical setting).
#
# Table 8 = JGA with trained retrievers (5% training data) on the full MultiWOZ
#          2.4 test set, comparing retrievers trained on Individual-scored vs
#          CombiSearch-scored data, with Llama-3-8B and 4-bit 70B as DST models.
#
# Pipeline shape: practical (trained-retriever) DST inference (run_dst / S4).
#
# Default: run the 4 DST configs only (retrievers/indexes assumed on disk).
# FULL=1 rebuilds: preprocess -> indexes -> data gen (CombiSearch + Individual,
# 5%) -> retriever training (combisearch/5% + individual/5%).
#
#   FULL=1 bash runner/table8.sh
#   DRY_RUN=1 bash runner/table8.sh
# =============================================================================

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

STAGES="${ROOT_DIR}/runner/stages"
FULL="${FULL:-0}"

# Data generation: CombiSearch-scored and Individual-scored data at 5%.
DATA_GEN_CONFIGS=(
  configs/data_generation/mwoz21/5p/split_v1.json
  configs/data_generation/mwoz21/5p/individual.json
)
# Retriever fine-tuning on each scored dataset.
TRAIN_CONFIGS=(
  configs/finetuning/mwoz21/combisearch/5p/split_v1.json
  configs/finetuning/mwoz21/individual/5p/split_v1.json
)
# DST inference — {8B,70B} x {combisearch,individual} (4 runs).
DST_CONFIGS=( configs/table_8/*/*/split_v*.json )

if [[ "${FULL}" == "1" ]]; then
  section "S0. Preprocess MultiWOZ"
  bash "${STAGES}/s0_preprocess.sh"

  section "S1. Build pretrained SBERT indexes"
  bash "${STAGES}/s1_build_indexes.sh"

  section "S2. Data generation (CombiSearch + Individual, 5%)"
  bash "${STAGES}/s2_generate_data.sh" "${DATA_GEN_CONFIGS[@]}"

  section "S3. Retriever fine-tuning (combisearch/5% + individual/5%)"
  bash "${STAGES}/s3_train_retriever.sh" "${TRAIN_CONFIGS[@]}"
fi

section "S4. DST inference — Table 8 (practical, 8B/70B)"
bash "${STAGES}/s4_run_dst.sh" "${DST_CONFIGS[@]}"

echo
echo "[done] Table 8 configs run. Check outputs/runs/table_8/**/running_log.json"
