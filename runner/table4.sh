#!/usr/bin/env bash
# =============================================================================
# Run the Table 4 DST configs (coreference; JSON vs Python prompt format).
#
# Table 4 = coreference-only evaluation with Llama-3-8B-Instruct, comparing the
#          JSON and Python prompt formats for CombiSearch and RefPyDST, in the
#          0% (zero-shot, single format example) and 5% (few-shot) settings.
#          Eval data is MultiWOZ 2.4 (coref dialogues selected via 2.3 annots).
#
# Pipeline shape: practical (trained-retriever) DST inference (run_dst / S4).
# The 0% configs use no retriever; the 5% configs use MultiWOZ combisearch /
# refpydst 5% retrievers.
#
# Default: run the 14 DST configs only.
# FULL=1 rebuilds: preprocess -> indexes -> CombiSearch data gen (5%) ->
# CombiSearch retriever training (5%) -> download RefPyDST 5% (v1/v2/v3).
#
#   FULL=1 bash runner/table4.sh
#   DRY_RUN=1 bash runner/table4.sh
# =============================================================================

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

STAGES="${ROOT_DIR}/runner/stages"
FULL="${FULL:-0}"

# CombiSearch data generation + retriever fine-tuning for the 5% setting.
COMBISEARCH_DATA_GEN_CONFIGS=(
  configs/data_generation/mwoz21/5p/split_v1.json
)
COMBISEARCH_TRAIN_CONFIGS=(
  configs/finetuning/mwoz21/combisearch/5p/split_v1.json
)
# DST inference — {0p,5p} x {combisearch,refpydst} x {json,python} (14 runs).
DST_CONFIGS=( configs/table_4/*/*/split_v*.json )

if [[ "${FULL}" == "1" ]]; then
  section "S0. Preprocess MultiWOZ (+ coref dialogues)"
  bash "${STAGES}/s0_preprocess.sh"

  section "S1. Build pretrained SBERT indexes"
  bash "${STAGES}/s1_build_indexes.sh"

  section "S2. CombiSearch data generation (5%)"
  bash "${STAGES}/s2_generate_data.sh" "${COMBISEARCH_DATA_GEN_CONFIGS[@]}"

  section "S3. CombiSearch retriever fine-tuning (5%)"
  bash "${STAGES}/s3_train_retriever.sh" "${COMBISEARCH_TRAIN_CONFIGS[@]}"

  section "S3-alt. Download RefPyDST retrievers (MultiWOZ 5% v1/v2/v3)"
  bash "${STAGES}/download_refpydst_retrievers.sh"
fi

section "S4. DST inference — Table 4 (coref, JSON vs Python)"
bash "${STAGES}/s4_run_dst.sh" "${DST_CONFIGS[@]}"

# Coreference-slot accuracy (the Table 4 metric) is computed per cell over its
# seeds (0p: 1 run; 5p: split_v1/v2/v3) from the S4 running_log.json outputs.
section "S5. Coreference-slot evaluation — Table 4"
TABLE4_RUNS="${COMBISEARCH_OUTPUTS_DIR}/runs/table_4"
for cell in "${TABLE4_RUNS}"/0p/* "${TABLE4_RUNS}"/5p/*; do
  [[ -d "${cell}" ]] || continue
  cell_logs=()
  for log in "${cell}"/split_v*/running_log.json; do
    [[ -f "${log}" ]] && cell_logs+=("${log}")
  done
  if [[ ${#cell_logs[@]} -eq 0 ]]; then
    echo "  [skip] no running_log.json under ${cell} (run S4 first)"
    continue
  fi
  echo "-- coref eval: $(basename "$(dirname "${cell}")")/$(basename "${cell}") (${#cell_logs[@]} run(s)) --"
  run "${PYTHON_BIN}" -m combisearch.evaluation.eval_coref_turns "${cell_logs[@]}"
done

echo
echo "[done] Table 4 configs run. Check outputs/runs/table_4/**/running_log.json"
