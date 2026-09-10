#!/bin/bash
# Download the MIND benchmark GT dataset (first-person + third-person) from HF
# (CSU-JPG/MIND) via the venv's huggingface_hub (no hf CLI needed). Lands under
# ../MIND-Data/, a sibling of this repository, which is what run_mind.sh sets
# as gt_root and what the drive_* scripts resolve:
#   ../MIND-Data/{perspective}/test/{test_type}/...
# This previously landed in ./mind_data, which nothing reads -- running both
# this and `src/download_models.py --dataset` produced two 35 GB copies.
# HF_TOKEN read from env if gated. Skips already-downloaded files (resumable).
#
#   bash download_mind_dataset.sh                 full dataset (both perspectives)
#   bash download_mind_dataset.sh first_person    only that subfolder
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"
DEST="$(realpath -m "$HERE/../MIND-Data")"
SUB="${1:-}"

if [ ! -x "$PY" ]; then
  echo "ERROR: $PY missing -- run setup_mind_venv.sh first" >&2
  exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "NOTE: HF_TOKEN not set (ok if public)"
fi

echo "Downloading CSU-JPG/MIND ${SUB} -> $DEST"
"$PY" -c "
import sys
from huggingface_hub import snapshot_download
sub = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
p = snapshot_download('CSU-JPG/MIND', repo_type='dataset', local_dir=r'$DEST',
                       allow_patterns=[sub + '/*'] if sub else None)
print('downloaded ->', p)
" "$SUB"

echo
echo "Done. gt_root for process.py = $DEST"
