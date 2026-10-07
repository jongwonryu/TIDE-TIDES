#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
MODEL="${1:-tides}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODEL" in
  tide) TRAINER="train/train_bulk_interface.py" ;;
  tides) TRAINER="train/train_interaction_aware.py" ;;
  *) echo "Usage: bash bash/train.sh {tide|tides} [training options]" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python "$TRAINER" --config "configs/${MODEL}_train.json" "$@"
