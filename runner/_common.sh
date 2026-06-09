# Shared environment and helpers for runner/ scripts.
# Source this from every runner/*.sh and runner/stages/*.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN="${DRY_RUN:-0}"

# Default to the project venv if present; honor an explicit PYTHON_BIN override.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

export COMBISEARCH_DATA_DIR="${COMBISEARCH_DATA_DIR:-${ROOT_DIR}/data}"
export COMBISEARCH_OUTPUTS_DIR="${COMBISEARCH_OUTPUTS_DIR:-${ROOT_DIR}/outputs}"
export COMBISEARCH_CONFIG_ROOT="${COMBISEARCH_CONFIG_ROOT:-${ROOT_DIR}/configs}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-combisearch}"

# Print "+ <cmd>" then run, unless DRY_RUN=1.
run() {
  printf '+ %s\n' "$*"
  printf ''
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

# Banner for a high-level section.
section() {
  echo
  echo "=============================================================="
  echo " $*"
  echo "=============================================================="
}
