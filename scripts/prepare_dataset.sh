#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
python -m src.data.dataset_preparation --config configs/dataset.yaml
