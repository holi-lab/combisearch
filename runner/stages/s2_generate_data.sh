#!/usr/bin/env bash
# Generate D_CombiSearch (scored example pools) for each given config.
#
# Usage:
#   bash runner/stages/s2_generate_data.sh <config.json> [<config.json> ...]

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

for cfg in "$@"; do
  echo "[generate_data] ${cfg}"
  run "${PYTHON_BIN}" "${ROOT_DIR}/src/combisearch/cli/generate_data.py" "${cfg}"
done
