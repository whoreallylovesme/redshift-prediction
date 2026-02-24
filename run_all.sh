#!/usr/bin/env bash
set -euo pipefail

rm -rf prepared
rm -rf runs

echo ">> prepare_dataset"

python prepare_dataset.py \
  --input dsat.csv \
  --data-config configs/dataset.yaml \
  --features features/features.yaml \
  --outdir prepared

echo ">> prepare_dataset done"

echo ">> model training start"

for cfg in catboost gam llr mlp rf; do
  echo ">> ${cfg} set1"

  python train.py \
    --train prepared/train.parquet \
    --config configs/${cfg}.yaml \
    --features features/features.yaml \
    --feature-set set1 \
    --outdir runs/${cfg}_set1

  python inference.py \
    --model runs/${cfg}_set1/model.pkl \
    --test prepared/test.parquet \
    --features features/features.yaml \
    --feature-set set1 \
    --out runs/${cfg}_set1/preds.parquet

  python visualize.py \
    --preds runs/${cfg}_set1/preds.parquet \
    --outdir runs/${cfg}_set1
done

echo ">> train set1 done"

for cfg in catboost gam llr mlp rf; do
  echo ">> ${cfg} set2"

  python train.py \
    --train prepared/train.parquet \
    --config configs/${cfg}.yaml \
    --features features/features.yaml \
    --feature-set set2 \
    --outdir runs/${cfg}_set2

  python inference.py \
    --model runs/${cfg}_set2/model.pkl \
    --test prepared/test.parquet \
    --features features/features.yaml \
    --feature-set set2 \
    --out runs/${cfg}_set2/preds.parquet

  python visualize.py \
    --preds runs/${cfg}_set2/preds.parquet \
    --outdir runs/${cfg}_set2
done

echo ">> train set2 done"
