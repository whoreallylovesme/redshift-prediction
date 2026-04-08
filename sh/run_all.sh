#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

rm -rf prepared
rm -rf runs

echo ">> prepare_dataset"

uv run python runtime/prepare_dataset.py \
  --input "datasets/dsat.csv" \
  --data-config configs/dataset.yaml \
  --features features/features.yaml \
  --outdir prepared

echo ">> prepare_dataset done"

echo ">> model training start"

for cfg in catboost llr mlp rf; do
  echo ">> ${cfg} set2"

  uv run python runtime/train.py \
    --train prepared/train.parquet \
    --config configs/${cfg}.yaml \
    --features features/features.yaml \
    --feature-set set2 \
    --outdir runs/${cfg}_set2

  uv run python runtime/inference.py \
    --model runs/${cfg}_set2/model.pkl \
    --test prepared/test.parquet \
    --features features/features.yaml \
    --feature-set set2 \
    --out runs/${cfg}_set2/preds.parquet

  uv run python runtime/calc_metrics.py \
    --preds runs/${cfg}_set2/preds.parquet \
    --outdir runs/${cfg}_set2 \
    --model-name ${cfg}
done

uv run python runtime/table.py \
  --runs-dir runs \
  --out "runs/summary_table.html" \
  --title "main"

uv run python runtime/plots_html.py \
  --runs-dir runs \
  --out "runs/summary_plots.html" \
  --title "main"

echo ">> train set2 done"
