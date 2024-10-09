#!/usr/bin/env bash
set -euo pipefail

# To overcome over-fitting, we fine-tuned the hyperparameter settings in the few-shot regime.
# To overcome over-smoothing, we fine-tuned the hyperparameter settings when the coarsening ratio is 0.1.
# The examples below use a coarsening ratio of 0.5.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

uv run python train.py --dataset cora --experiment fixed --coarsening_ratio 0.5
uv run python train.py --dataset cora --experiment few --epochs 100 --coarsening_ratio 0.5
uv run python train.py --dataset citeseer --experiment fixed --epochs 200 --coarsening_ratio 0.5
uv run python train.py --dataset pubmed --experiment fixed --epochs 200 --coarsening_ratio 0.5
uv run python train.py --dataset pubmed --experiment few --epochs 60 --coarsening_ratio 0.5
uv run python train.py --dataset dblp --experiment random --epochs 50 --coarsening_ratio 0.5
uv run python train.py --dataset Physics --experiment random --epochs 200 --lr 0.001 --weight_decay 0 --coarsening_ratio 0.5
