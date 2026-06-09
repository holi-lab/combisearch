#!/usr/bin/env bash
# Download RefPyDST's published fine-tuned retrievers from HuggingFace
# (Brendan/refpydst-*-referredstates-split-v*) directly into the canonical
# combisearch retriever layout that the table_* inference configs expect:
#
#   outputs/retrievers/finetuning/mwoz21/refpydst/<ratio>/split_v<n>/
#
# That way `bash runner/table1.sh` works without any post-download mv.

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

echo "[download_refpydst_retrievers] target = outputs/retrievers/finetuning/mwoz21/refpydst/"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "+ huggingface_hub.snapshot_download for 1p,5p,10p × v1,v2,v3 and 100p"
  exit 0
fi

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(os.environ["COMBISEARCH_OUTPUTS_DIR"]) / "retrievers" / "finetuning" / "mwoz21" / "refpydst"

# Few-shot retrievers: 1p / 5p / 10p × split_v1 / v2 / v3
for ratio in ("1p", "5p", "10p"):
    for v in ("v1", "v2", "v3"):
        repo = f"Brendan/refpydst-{ratio}-referredstates-split-{v}"
        dst  = ROOT / ratio / f"split_{v}"
        print(f"  {repo} -> {dst}")
        snapshot_download(repo_id=repo, local_dir=str(dst), local_dir_use_symlinks=False)

# Full-data retriever (single split — landed at split_v1/ to match the few-shot layout).
repo = "Brendan/refpydst-100p-referredstates"
dst  = ROOT / "100p" / "split_v1"
print(f"  {repo} -> {dst}")
snapshot_download(repo_id=repo, local_dir=str(dst), local_dir_use_symlinks=False)
PY
