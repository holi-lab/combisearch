#!/usr/bin/env bash
# =============================================================================
# Run the Table 5 configs (oracle upper bound).
#
# Table 5 = oracle JGA (gold-state access, examples selected directly) for
#          RefPyDST / Hybrid / CombiSearch scoring with Llama-3-8B and 4-bit
#          70B, at 1% / 5% / 100%. Uses the pretrained mpnet SBERT retriever
#          (no fine-tuned retriever).
#
# Pipeline split (by classify_config):
#   * combisearch/* carry num_sampling_iteration -> data_generation (S2): the
#     combinatorial CombiSearch scoring runs that produce the oracle JGA.
#   * refpydst/* and hybrid/* are single-inference baselines -> run_dst (S4).
#
# Default: run the experiment configs (pretrained index assumed on disk).
# FULL=1 rebuilds the prerequisites only: preprocess + build pretrained index
# (no retriever training — the oracle setting uses the pretrained SBERT).
#
#   FULL=1 bash runner/table5.sh
#   DRY_RUN=1 bash runner/table5.sh
# =============================================================================

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

STAGES="${ROOT_DIR}/runner/stages"
FULL="${FULL:-0}"

# Combinatorial CombiSearch scoring (oracle) -> generate_data (18 runs).
GENERATE_DATA_CONFIGS=( configs/table_5/combisearch/*/*/split_v*.json )
# Single-inference baselines RefPyDST + Hybrid -> run_dst (36 runs).
RUN_DST_CONFIGS=( configs/table_5/refpydst/*/*/split_v*.json
                  configs/table_5/hybrid/*/*/split_v*.json )

if [[ "${FULL}" == "1" ]]; then
  section "S0. Preprocess MultiWOZ"
  bash "${STAGES}/s0_preprocess.sh"

  section "S1. Build pretrained SBERT indexes"
  bash "${STAGES}/s1_build_indexes.sh"
fi

section "S2. CombiSearch oracle scoring (combisearch rows)"
bash "${STAGES}/s2_generate_data.sh" "${GENERATE_DATA_CONFIGS[@]}"

section "S4. Oracle single-inference baselines (refpydst + hybrid rows)"
bash "${STAGES}/s4_run_dst.sh" "${RUN_DST_CONFIGS[@]}"

echo
echo "[done] Table 5 configs run. Check outputs/runs/table_5/**/running_log.json"
