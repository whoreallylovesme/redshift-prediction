set -euo pipefail

echo ">> Teddy pipeline run"

rm -rf runs_B runs_C runs_D

echo "===TEDDY==="

for cfg in catboost llr mlp rf; do
  echo ">> model training start (${cfg})"

  tmp_train_dir="tmp_${cfg}"
  rm -rf "$tmp_train_dir"

  python train.py \
    --train TeddyParquet/teddy_A.parquet \
    --config "configs/${cfg}.yaml" \
    --features features/features_teddy_happy.yaml \
    --feature-set set3 \
    --outdir "$tmp_train_dir"

  echo ">> model inference start (${cfg})"

  for suf in B C D; do
    dsat="teddy_${suf}"
    run_dir="runs_${suf}/${cfg}"
    mkdir -p "$run_dir"

    preds="${run_dir}/preds.parquet"

    python inference.py \
      --model "${tmp_train_dir}/model.pkl" \
      --test "TeddyParquet/${dsat}.parquet" \
      --features features/features_teddy_happy.yaml \
      --feature-set set3 \
      --out "$preds"

    python visualize.py \
      --preds "$preds" \
      --outdir "$run_dir" \
      --model-name "${cfg}_${dsat}"
  done

  rm -rf "$tmp_train_dir"
done

for x in runs_B runs_C runs_D; do
    python table.py \
        --runs-dir ${x} \
        --out ${x}/summary.html \
        --title ${x}
done

echo ">> done"
