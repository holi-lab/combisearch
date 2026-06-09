#!/usr/bin/env bash
# =============================================================================
# Run the Table 3 DST configs (SGD / cross-dataset).
#
# Table 3 = JGA on the SGD test set with Llama-3-8B-Instruct, for Random /
#          RefPyDST / CombiSearch retrievers, trained either in-domain ("SGD",
#          5% of SGD) or cross-dataset ("MultiWOZ", 5% of MultiWOZ 2.1).
#          NOTE: SGD retrieval is dense-only (SBERT), not the BM25+SBERT hybrid.
#
# Pipeline shape: practical (trained-retriever) DST inference (run_dst / S4).
#
# Default: run the 6 DST configs only.
# FULL=1 rebuilds: preprocess -> indexes -> data gen (SGD + MultiWOZ 5%) ->
# retriever training (SGD combisearch/refpydst + MultiWOZ combisearch 5%) ->
# download MultiWOZ RefPyDST 5% retriever. (Random uses no retriever.)
#
#   FULL=1 bash runner/table3.sh
#   DRY_RUN=1 bash runner/table3.sh
# =============================================================================

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

STAGES="${ROOT_DIR}/runner/stages"
FULL="${FULL:-0}"

# Data generation feeding the trained retrievers.
DATA_GEN_CONFIGS=(
  configs/data_generation/sgd/split_v1.json
  configs/data_generation/mwoz21/5p/split_v1.json
)
# Retriever fine-tuning. (MultiWOZ RefPyDST 5% comes from the download stage;
# SGD RefPyDST has no HF download, so it is trained here.)
TRAIN_CONFIGS=(
  configs/finetuning/sgd/combisearch.json
  configs/finetuning/sgd/refpydst.json
  configs/finetuning/mwoz21/combisearch/5p/split_v1.json
)
# DST inference — {mwoz, sgd} x {combisearch, random, refpydst} (6 runs).
DST_CONFIGS=( configs/table_3/*/*.json )

if [[ "${FULL}" == "1" ]]; then
  section "S0. Preprocess MultiWOZ + SGD"
  bash "${STAGES}/s0_preprocess.sh"

  section "S1. Build pretrained SBERT indexes"
  bash "${STAGES}/s1_build_indexes.sh"

  section "S2. Data generation (SGD + MultiWOZ 5%)"
  bash "${STAGES}/s2_generate_data.sh" "${DATA_GEN_CONFIGS[@]}"

  section "S3. Retriever fine-tuning (SGD combisearch/refpydst + MultiWOZ combisearch 5%)"
  bash "${STAGES}/s3_train_retriever.sh" "${TRAIN_CONFIGS[@]}"

  section "S3-alt. Download RefPyDST retrievers (MultiWOZ 5%)"
  bash "${STAGES}/download_refpydst_retrievers.sh"
fi

section "S4. DST inference — Table 3 (SGD / cross-dataset)"
bash "${STAGES}/s4_run_dst.sh" "${DST_CONFIGS[@]}"

echo
echo "[done] Table 3 configs run. Check outputs/runs/table_3/**/running_log.json"
