#!/usr/bin/env bash
# Build pretrained SBERT dense indexes used by CombiSearch data generation
# and by retrievers that fall back to the unmodified SBERT encoder.
#
# Pass --only <name> flags through to filter which indexes are built.
# Without args, builds all specs declared in src/combisearch/indexing/specs.py.

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

run "${PYTHON_BIN}" "${ROOT_DIR}/src/combisearch/cli/build_indexes.py" "$@"
