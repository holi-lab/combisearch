#!/usr/bin/env bash
# =============================================================================
# Reproduce Table 1.
#
# Table 1 = JGA on MultiWOZ 2.1 / 2.4 with gpt-3.5-turbo, comparing the
#          CombiSearch retriever against the RefPyDST baseline, at 1% / 5%
#          / 100% of the training data (mean over splits v1/v2/v3 for the
#          few-shot rows, single split for 100%).
#
# Default behavior: only run DST inference (28 configs total). All retrievers
# and indexes are assumed to already be on disk.
#
# To rebuild everything from scratch (preprocess → indexes → CombiSearch
# data gen → CombiSearch retriever training → download RefPyDST retrievers →
# DST inference), set FULL=1:
#
#   FULL=1 OPENAI_API_KEY=... bash runner/table1.sh
#
# Dry run (prints commands only):
#
#   DRY_RUN=1 OPENAI_API_KEY=dummy bash runner/table1.sh
# =============================================================================

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

STAGES="${ROOT_DIR}/runner/stages"
FULL="${FULL:-0}"

# -----------------------------------------------------------------------------
# Configs that compose Table 1.
# Listed explicitly so it is obvious which 28 DST runs make up the table and
# so individual rows can be commented out without reasoning about globs.
# -----------------------------------------------------------------------------

# CombiSearch data generation: one pass per training ratio (split_v1 only).
COMBISEARCH_DATA_GEN_CONFIGS=(
  configs/data_generation/mwoz21/1p/split_v1.json
  configs/data_generation/mwoz21/5p/split_v1.json
  configs/data_generation/mwoz21/100p/split_v1.json
)

# CombiSearch retriever fine-tuning: one retriever per training ratio.
COMBISEARCH_TRAIN_CONFIGS=(
  configs/finetuning/mwoz21/combisearch/1p/split_v1.json
  configs/finetuning/mwoz21/combisearch/5p/split_v1.json
  configs/finetuning/mwoz21/combisearch/100p/split_v1.json
)

# DST inference — CombiSearch retriever × gpt-3.5-turbo (14 runs).
COMBISEARCH_DST_CONFIGS=(
  configs/table_1/mw21/combisearch/1p/split_v1.json
  configs/table_1/mw21/combisearch/1p/split_v2.json
  configs/table_1/mw21/combisearch/1p/split_v3.json
  configs/table_1/mw21/combisearch/5p/split_v1.json
  configs/table_1/mw21/combisearch/5p/split_v2.json
  configs/table_1/mw21/combisearch/5p/split_v3.json
  configs/table_1/mw21/combisearch/100p/split_v1.json
  configs/table_1/mw24/combisearch/1p/split_v1.json
  configs/table_1/mw24/combisearch/1p/split_v2.json
  configs/table_1/mw24/combisearch/1p/split_v3.json
  configs/table_1/mw24/combisearch/5p/split_v1.json
  configs/table_1/mw24/combisearch/5p/split_v2.json
  configs/table_1/mw24/combisearch/5p/split_v3.json
  configs/table_1/mw24/combisearch/100p/split_v1.json
)

# DST inference — RefPyDST baseline × gpt-3.5-turbo (14 runs).
REFPYDST_DST_CONFIGS=(
  configs/table_1/mw21/refpydst/1p/split_v1.json
  configs/table_1/mw21/refpydst/1p/split_v2.json
  configs/table_1/mw21/refpydst/1p/split_v3.json
  configs/table_1/mw21/refpydst/5p/split_v1.json
  configs/table_1/mw21/refpydst/5p/split_v2.json
  configs/table_1/mw21/refpydst/5p/split_v3.json
  configs/table_1/mw21/refpydst/100p/split_v1.json
  configs/table_1/mw24/refpydst/1p/split_v1.json
  configs/table_1/mw24/refpydst/1p/split_v2.json
  configs/table_1/mw24/refpydst/1p/split_v3.json
  configs/table_1/mw24/refpydst/5p/split_v1.json
  configs/table_1/mw24/refpydst/5p/split_v2.json
  configs/table_1/mw24/refpydst/5p/split_v3.json
  configs/table_1/mw24/refpydst/100p/split_v1.json
)

# -----------------------------------------------------------------------------
# Optional: full pipeline from raw data.
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Always: DST inference (the only thing strictly needed to reproduce Table 1
# numbers if you already have the retrievers).
# -----------------------------------------------------------------------------
section "S4. DST inference — CombiSearch retriever × gpt-3.5-turbo"
bash "${STAGES}/s4_run_dst.sh" "${COMBISEARCH_DST_CONFIGS[@]}"

section "S4. DST inference — RefPyDST baseline × gpt-3.5-turbo"
bash "${STAGES}/s4_run_dst.sh" "${REFPYDST_DST_CONFIGS[@]}"

echo
echo "[done] Table 1 finished. Check outputs/runs/table_1/**/running_log.json"
