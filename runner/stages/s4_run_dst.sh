#!/usr/bin/env bash
# Run DST inference (combisearch.cli.run_dst) for each given config.
#
# Usage:
#   bash runner/stages/s4_run_dst.sh <config.json> [<config.json> ...]

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

for cfg in "$@"; do
  echo "[run_dst] ${cfg}"
  run "${PYTHON_BIN}" "${ROOT_DIR}/src/combisearch/cli/run_dst.py" "${cfg}"
done
