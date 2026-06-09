#!/usr/bin/env bash
# =============================================================================
# S0 — Preprocess datasets into data.
#
# Builds MultiWOZ splits, MW2.3 coref data, and SGD-derived files.
# =============================================================================

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${ROOT_DIR}"

mkdir -p data/raw

log() {
  printf '[S0] %s\n' "$*"
}

subsection() {
  printf '\n[S0] %s\n\n' "$*"
}

section "S0 | Preprocess raw corpora"
log "Raw files:       data/raw"


# 1. MultiWOZ 2.1/2.4 dialogue files.
section "1/8 | MultiWOZ 2.1 and 2.4"
log "Converting raw JSON into train/dev/test dialogue files."
echo
run "${PYTHON_BIN}" data/code/download_create_data.py --main_dir data/raw/mwz21 --mwz_ver 2.1 --target_path data/raw/mwz2.1
run "${PYTHON_BIN}" data/code/download_create_data.py --main_dir data/raw/mwz24 --mwz_ver 2.4 --target_path data/raw/mwz2.4


# 2. MultiWOZ 2.1 sampled train splits.
section "2/8 | MultiWOZ 2.1 sampled train splits"
log "Sampling 1%, 5%, and 10% train sets across seeds 88, 42, and 888."

subsection "MultiWOZ 2.1 train | 1%"
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.1/train_dials.json --target_fn data/mw21_1p_train_v1.json --ratio 0.01 --seed 88
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.1/train_dials.json --target_fn data/mw21_1p_train_v2.json --ratio 0.01 --seed 42
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.1/train_dials.json --target_fn data/mw21_1p_train_v3.json --ratio 0.01 --seed 888

subsection "MultiWOZ 2.1 train | 5%"
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.1/train_dials.json --target_fn data/mw21_5p_train_v1.json --ratio 0.05 --seed 88
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.1/train_dials.json --target_fn data/mw21_5p_train_v2.json --ratio 0.05 --seed 42
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.1/train_dials.json --target_fn data/mw21_5p_train_v3.json --ratio 0.05 --seed 888

subsection "MultiWOZ 2.1 train | 10%"
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.1/train_dials.json --target_fn data/mw21_10p_train_v1.json --ratio 0.1 --seed 88
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.1/train_dials.json --target_fn data/mw21_10p_train_v2.json --ratio 0.1 --seed 42
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.1/train_dials.json --target_fn data/mw21_10p_train_v3.json --ratio 0.1 --seed 888

# 3. MultiWOZ 2.1 full splits.
section "3/8 | MultiWOZ 2.1 full corpus"
log "Creating full train/dev/test files."
echo
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.1/train_dials.json --target_fn data/mw21_100p_train.json --ratio 1.0
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.1/dev_dials.json   --target_fn data/mw21_100p_dev.json   --ratio 1.0
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.1/test_dials.json  --target_fn data/mw21_100p_test.json  --ratio 1.0

# 4. MultiWOZ 2.4 eval splits.
section "4/8 | MultiWOZ 2.4 dev/test subsets"
log "Sampling dev/test subsets used for evaluation."

subsection "MultiWOZ 2.4 dev | 5%, 10%, 20%"
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.4/dev_dials.json  --target_fn data/mw24_5p_dev.json      --ratio 0.05 --seed 88
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.4/dev_dials.json  --target_fn data/mw24_10p_dev.json     --ratio 0.1  --seed 88
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.4/dev_dials.json  --target_fn data/mw24_20p_dev.json     --ratio 0.2  --seed 42

subsection "MultiWOZ 2.4 dev/test | 100%"
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.4/dev_dials.json  --target_fn data/mw24_100p_dev.json    --ratio 1.0
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.4/test_dials.json --target_fn data/mw24_100p_test.json   --ratio 1.0

subsection "MultiWOZ 2.4 test | 10% across 3 seeds"
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.4/test_dials.json --target_fn data/mw24_10p_test_v1.json --ratio 0.1 --seed 88
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.4/test_dials.json --target_fn data/mw24_10p_test_v2.json --ratio 0.1 --seed 42
run "${PYTHON_BIN}" data/code/sample_by_ratio.py --input_fn data/raw/mwz2.4/test_dials.json --target_fn data/mw24_10p_test_v3.json --ratio 0.1 --seed 888


# 5. MultiWOZ 2.3 coref annotations.

section "5/8 | MultiWOZ 2.3 coreference data"
if [[ -f "data/raw/mw23/data.json" ]]; then
  log "Skipping download: data/raw/mw23/data.json already present."
elif [[ "${DRY_RUN}" == "1" ]]; then
  log "Dry run: would download MultiWOZ 2.3 coref zip."
  echo
  echo "+ (dry-run) curl -L https://github.com/lexmen318/MultiWOZ-coref/raw/main/MultiWOZ2_3.zip -> data/raw/mw23/data.json"
else
  mkdir -p data/raw/mw23
  tmpdir="$(mktemp -d)"
  log "Downloading MultiWOZ 2.3 coref zip."
  echo
  curl -L -o "${tmpdir}/MultiWOZ2_3.zip" \
      "https://github.com/lexmen318/MultiWOZ-coref/raw/main/MultiWOZ2_3.zip"
  unzip -q "${tmpdir}/MultiWOZ2_3.zip" -d "${tmpdir}"
  mv "${tmpdir}/MultiWOZ2_3/data.json" data/raw/mw23/data.json
  rm -rf "${tmpdir}"
fi

# 6. Coreference-only dev set.
section "6/8 | Coreference-only dev set"
if [[ -f "data/raw/mw23/data.json" ]] && [[ -f "data/mw24_100p_dev.json" ]]; then
  log "Building coreference-only dev set for Table 4."
  echo
  run "${PYTHON_BIN}" data/code/build_coref_only_dataset.py
else
  log "Skipping build: needs data/raw/mw23/data.json and data/mw24_100p_dev.json."
fi

# 7. Schema-Guided Dialogue download.
section "7/8 | Schema-Guided Dialogue download"
if [[ -d "data/raw/sgd/SGD/train" ]]; then
  log "Skipping clone: data/raw/sgd/SGD already present."
elif [[ "${DRY_RUN}" == "1" ]]; then
  log "Dry run: would clone Schema-Guided Dialogue repository."
  echo
  echo "+ (dry-run) git clone --depth 1 https://github.com/google-research-datasets/dstc8-schema-guided-dialogue data/raw/sgd/SGD"
else
  mkdir -p data/raw/sgd
  log "Cloning Schema-Guided Dialogue (SGD) repository."
  echo
  git clone --depth 1 \
      https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git \
      data/raw/sgd/SGD
fi


# 8. SGD → MultiWOZ-delta.

section "8/8 | Schema-Guided Dialogue conversion"
SGD_DIR="${SGD_DATA_DIR:-}"
if [[ -z "${SGD_DIR}" ]]; then
  for candidate in data/raw/sgd/SGD data/sgd/SGD data/SGD data/sgd; do
    if [[ -d "${candidate}" ]]; then
      SGD_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${SGD_DIR}" ]]; then
  log "Skipping preprocessing: SGD raw not found. Set SGD_DATA_DIR=<path> or re-run after the download step succeeds."
else
  log "Converting SGD to MultiWOZ-delta format for Table 3."
  echo
  run "${PYTHON_BIN}" data/code/sgd_to_multiwoz_delta_preprocess.py \
      --sgd-dir "${SGD_DIR}" \
      --output-dir data/ \
      --ids-dir data/sgd_ids
fi

section "S0 complete"
log "Processed files are under data."
