#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    uv run python train.py \
        --dataset citeseer \
        --coarsening_method kron \
        --experiment fixed \
        --coarsening_ratio "$ratio"
done
