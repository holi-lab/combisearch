#!/usr/bin/env bash
# =============================================================================
# Download and preprocess every dataset the experiments consume.
#
# Builds MultiWOZ 2.1 / 2.4 splits, MultiWOZ 2.3 coref data, and the
# SGD-derived files into data/. This is the S0 stage, exposed as a top-level
# entry point alongside runner/table1.sh.
#
# Dry run (declares actions only, no downloads):
#   DRY_RUN=1 bash runner/preprocess.sh
#
# If the SGD corpus is already on disk elsewhere:
#   SGD_DATA_DIR=/path/to/SGD bash runner/preprocess.sh
# =============================================================================

set -Eeuo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/stages/s0_preprocess.sh" "$@"
