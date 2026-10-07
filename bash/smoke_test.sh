#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
python scripts/smoke_test.py "$@"
