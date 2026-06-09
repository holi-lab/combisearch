#!/usr/bin/env bash
# =============================================================================
# Run the Table 7 configs (individual vs combinatorial scoring, oracle).
#
# Table 7 = oracle JGA with Llama-3-8B over 100 candidates drawn from the full
#          MultiWOZ training set: Individual (1 evaluation/example), CombiSearch
#          M=3, and CombiSearch M=9. Uses the pretrained mpnet SBERT retriever.
#
# Pipeline shape: all three rows are scoring runs (num_sampling_iteration set:
#   individual=1, 3iter=3, 9iter=9) -> data_generation (S2).
#
# Default: run the 9 scoring configs (pretrained index assumed on disk).
# FULL=1 rebuilds prerequisites only: preprocess + build pretrained index.
#
#   FULL=1 bash runner/table7.sh
#   DRY_RUN=1 bash runner/table7.sh
# =============================================================================

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

STAGES="${ROOT_DIR}/runner/stages"
FULL="${FULL:-0}"

# Individual (M=1), CombiSearch M=3, CombiSearch M=9 -> generate_data (9 runs).
GENERATE_DATA_CONFIGS=( configs/table_7/*/split_v*.json )

if [[ "${FULL}" == "1" ]]; then
  section "S0. Preprocess MultiWOZ"
  bash "${STAGES}/s0_preprocess.sh"

  section "S1. Build pretrained SBERT indexes"
  bash "${STAGES}/s1_build_indexes.sh"
fi

section "S2. Individual / CombiSearch M=3 / M=9 oracle scoring"
bash "${STAGES}/s2_generate_data.sh" "${GENERATE_DATA_CONFIGS[@]}"

echo
echo "[done] Table 7 configs run. Check outputs/runs/table_7/**/running_log.json"
