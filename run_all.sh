set -euo pipefail

rm -rf prepared

python prepare_dataset.py --input dsat.csv --features features/features.yaml --outdir prepared

echo ">> prepare_dataset done"

rm -rf runs

for cfg in catboost gam llr mlp rf; do
  echo ">> training->"${cfg} && \
  python train.py \
    --dataset prepared/data.parquet \
    --config configs/${cfg}.yaml \
    --features features/features.yaml \
    --feature-set set1 \
    --outdir runs/${cfg}_set1
done

echo ">> train done"
