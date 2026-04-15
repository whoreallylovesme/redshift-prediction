import shutil

import click
from prefect import flow, task

from runtime.calc_metrics import calc_metrics
from runtime.inference import inference
from runtime.plots_html import summary_plots
from runtime.prepare_dataset import prepare
from runtime.table import summary_table
from runtime.train import train


@task
def prepare_dataset_task(
    input_csv: str, data_config: str, features: str, outdir: str,
) -> None:
    prepare(input_csv, data_config, features, outdir)


@task
def train_task(
    config: str, outdir: str,
    train_path: str | None = None, features: str | None = None, feature_set: str | None = None,
) -> None:
    train(config, outdir, train_path, features, feature_set)


@task
def inference_task(
    model_path: str, out: str,
    config: str | None = None, test_path: str | None = None,
    features: str | None = None, feature_set: str | None = None,
    data_dir: str | None = None, catalog: str | None = None,
) -> None:
    inference(model_path, out, config, test_path, features, feature_set, data_dir, catalog)


@task
def calc_metrics_task(preds: str, outdir: str, model_name: str) -> None:
    calc_metrics(preds, outdir, model_name)


@task
def summary_table_task(runs_dir: str, out: str, title: str) -> None:
    summary_table(runs_dir, out, title)


@task
def summary_plots_task(runs_dir: str, out: str, title: str) -> None:
    summary_plots(runs_dir, out, title)


@flow(name="fast-test")
def fast_test() -> None:
    shutil.rmtree("prepared", ignore_errors=True)
    shutil.rmtree("runs", ignore_errors=True)

    prepare_dataset_task(
        "datasets/dsat.csv", "configs/dataset.yaml", "features/features.yaml", "prepared",
    )

    for cfg in ["catboost_fast_test"]:
        outdir = f"runs/{cfg}_set1"
        train_task(
            f"configs/{cfg}.yaml", outdir,
            train_path="prepared/train.parquet",
            features="features/features.yaml",
            feature_set="set1",
        )
        inference_task(
            f"{outdir}/model.pkl", f"{outdir}/preds.parquet",
            test_path="prepared/test.parquet",
            features="features/features.yaml",
            feature_set="set1",
        )
        calc_metrics_task(f"{outdir}/preds.parquet", outdir, cfg)

    summary_table_task("runs", "runs/summary_table.html", "main")
    summary_plots_task("runs", "runs/summary_plots.html", "main")


@flow(name="run-all")
def run_all() -> None:
    shutil.rmtree("prepared", ignore_errors=True)
    shutil.rmtree("runs", ignore_errors=True)

    prepare_dataset_task(
        "datasets/dsat.csv", "configs/dataset.yaml", "features/features.yaml", "prepared",
    )

    for cfg in ["catboost", "llr", "mlp", "rf"]:
        outdir = f"runs/{cfg}_set2"
        train_task(
            f"configs/{cfg}.yaml", outdir,
            train_path="prepared/train.parquet",
            features="features/features.yaml",
            feature_set="set2",
        )
        inference_task(
            f"{outdir}/model.pkl", f"{outdir}/preds.parquet",
            test_path="prepared/test.parquet",
            features="features/features.yaml",
            feature_set="set2",
        )
        calc_metrics_task(f"{outdir}/preds.parquet", outdir, cfg)

    summary_table_task("runs", "runs/summary_table.html", "main")
    summary_plots_task("runs", "runs/summary_plots.html", "main")


@flow(name="train-nn")
def train_nn() -> None:
    splits = ["B", "C", "D"]
    models = ["efficientnet_b0", "vit_tiny"]
    runs_root = "nn_runs"

    shutil.rmtree(runs_root, ignore_errors=True)

    for model_name in models:
        config_path = f"configs/{model_name}.yaml"
        model_dir = f"{runs_root}/{model_name}"

        train_task(config_path, model_dir)

        for split in splits:
            data_dir = f"teddy{split}_imgs"
            catalog = f"datasets/teddy_happy_orig/Teddy/teddy_{split}"
            split_dir = f"{runs_root}/runs_{split}/{model_name}"
            preds_path = f"{split_dir}/preds.parquet"

            inference_task(
                f"{model_dir}/best_model.pt", preds_path,
                config=config_path,
                data_dir=data_dir,
                catalog=catalog,
            )
            calc_metrics_task(preds_path, split_dir, f"{model_name}_teddy{split}")

    for split in splits:
        split_runs = f"{runs_root}/runs_{split}"
        summary_table_task(split_runs, f"{split_runs}/summary_table.html", f"nn_runs_{split}")
        summary_plots_task(split_runs, f"{split_runs}/summary_plots.html", f"nn_runs_{split}")


FLOWS = {
    "fast-test": fast_test,
    "run-all": run_all,
    "train-nn": train_nn,
}


@click.command()
@click.argument("flow_name", type=click.Choice(list(FLOWS.keys())))
def main(flow_name: str) -> None:
    FLOWS[flow_name]()


if __name__ == "__main__":
    main()
