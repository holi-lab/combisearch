#!/usr/bin/env bash
# =============================================================================
# Run the Table 6 configs (pool-construction x combinatorial-search ablation).
#
# Table 6 = oracle JGA with Llama-3-8B, comparing pool construction
#          (Random / SBERT / BM25&SBERT) with vs without combinatorial scoring.
#          Uses the pretrained mpnet SBERT retriever.
#
# Pipeline split (by classify_config):
#   * pool_scoring/*  carry num_sampling_iteration -> data_generation (S2):
#     CombiSearch combinatorial scoring on each pool.
#   * pool_only/*     are single-inference (N/A scoring)  -> run_dst (S4).
#
# Default: run the experiment configs (pretrained index assumed on disk).
# FULL=1 rebuilds prerequisites only: preprocess + build pretrained index.
#
#   FULL=1 bash runner/table6.sh
#   DRY_RUN=1 bash runner/table6.sh
#
# train_fn is mw21_100p_train.json (the full pool). The Se2 row is an external
# baseline and has no config here.
# =============================================================================

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

STAGES="${ROOT_DIR}/runner/stages"
FULL="${FULL:-0}"

# Combinatorial CombiSearch scoring per pool -> generate_data (9 runs).
GENERATE_DATA_CONFIGS=( configs/table_6/pool_scoring/*/split_v*.json )
# Pool-only single-inference (no combinatorial search) -> run_dst (9 runs).
RUN_DST_CONFIGS=( configs/table_6/pool_only/*/split_v*.json )

if [[ "${FULL}" == "1" ]]; then
  section "S0. Preprocess MultiWOZ"
  bash "${STAGES}/s0_preprocess.sh"

  section "S1. Build pretrained SBERT indexes"
  bash "${STAGES}/s1_build_indexes.sh"
fi

section "S2. CombiSearch oracle scoring (pool_scoring rows)"
bash "${STAGES}/s2_generate_data.sh" "${GENERATE_DATA_CONFIGS[@]}"

section "S4. Pool-only single inference (pool_only rows)"
bash "${STAGES}/s4_run_dst.sh" "${RUN_DST_CONFIGS[@]}"

echo
echo "[done] Table 6 configs run. Check outputs/runs/table_6/**/running_log.json"
