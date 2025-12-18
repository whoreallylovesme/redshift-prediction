set -euo pipefail

rm -rf prepared

python prepare_dataset.py --input dsat.csv --features features/features.yaml --outdir prepared

echo ">> prepare_dataset done"

rm -rf runs

echo ">> model training start"

for cfg in catboost gam llr mlp rf; do
  echo ">> training->"${cfg} && \
  python train.py \
    --dataset prepared/data.parquet \
    --config configs/${cfg}.yaml \
    --features features/features.yaml \
    --feature-set set1 \
    --outdir runs/${cfg}_set1
done

echo ">> train set1 done"

for cfg in catboost gam llr mlp rf; do
  echo ">> training->"${cfg} && \
  python train.py \
    --dataset prepared/data.parquet \
    --config configs/${cfg}.yaml \
    --features features/features.yaml \
    --feature-set set2 \
    --outdir runs/${cfg}_set2
done

echo ">> train set2 done"
