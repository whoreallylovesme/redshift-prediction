import shutil

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


if __name__ == "__main__":
    fast_test()
