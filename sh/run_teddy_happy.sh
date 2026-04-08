#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo ">> Teddy + Happy pipeline run"

DATASETS=("teddy" "happy")
SPLITS=("B" "C" "D")
MODELS=("catboost" "llr" "mlp" "rf")

for ds in "${DATASETS[@]}"; do

  if [ "$ds" = "teddy" ]; then
    base="TeddyResult"
    parquet_dir="TeddyParquet"
  else
    base="HappyResult"
    parquet_dir="HappyParquet"
  fi

  echo "=== DATASET: ${ds} -> ${base} (parquet: ${parquet_dir}) ==="

  rm -rf "${base}/runs_B" "${base}/runs_C" "${base}/runs_D"
  mkdir -p "${base}"

  for cfg in "${MODELS[@]}"; do
    echo ">> model training start (${cfg}, ${ds})"

    tmp_train_dir="tmp_${ds}_${cfg}"
    rm -rf "$tmp_train_dir"

    uv run python runtime/train.py \
      --train "${parquet_dir}/${ds}_A.parquet" \
      --config "configs/${cfg}.yaml" \
      --features features/features_teddy_happy.yaml \
      --feature-set set3 \
      --outdir "$tmp_train_dir"

    echo ">> model inference start (${cfg}, ${ds})"

    for suf in "${SPLITS[@]}"; do
      dsat="${ds}_${suf}"
      run_dir="${base}/runs_${suf}/${cfg}"
      mkdir -p "$run_dir"

      preds="${run_dir}/preds.parquet"

      uv run python runtime/inference.py \
        --model "${tmp_train_dir}/model.pkl" \
        --test "${parquet_dir}/${dsat}.parquet" \
        --features features/features_teddy_happy.yaml \
        --feature-set set3 \
        --out "$preds"

      uv run python runtime/calc_metrics.py \
        --preds "$preds" \
        --outdir "$run_dir" \
        --model-name "${cfg}_${dsat}"
    done

    rm -rf "$tmp_train_dir"
  done

  for suf in "${SPLITS[@]}"; do
    x="${base}/runs_${suf}"

    uv run python runtime/table.py \
      --runs-dir "${x}" \
      --out "${x}/summary_table.html" \
      --title "runs_${suf}"

    uv run python runtime/plots_html.py \
      --runs-dir "${x}" \
      --out "${x}/summary_plots.html" \
      --title "runs_${suf}"
  done

done

echo ">> done"
