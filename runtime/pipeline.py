import shutil
from pathlib import Path
from typing import cast

import click
import pandas as pd
import yaml
from prefect import flow, task

from lib.config import FlowConfig
from lib.inference import inference
from lib.utils.utils import load_yaml_as
from runtime.evaluate import calc_metrics
from runtime.render_plots import summary_plots
from runtime.render_table import summary_table
from runtime.split_tabular import prepare
from runtime.train import train


@task
def split_tabular_data(
    input_csv: str, data_config: str, features: str, outdir: str,
) -> None:
    prepare(input_csv, data_config, features, outdir)

@task
def fit_model(
    config: str, outdir: str,
    train_path: str | None = None, features: str | None = None, feature_set: str | None = None,
) -> None:
    train(config, outdir, train_path, features, feature_set)

@task
def predict(
    model_path: str, out: str,
    config: str | None = None, test_path: str | None = None,
    features: str | None = None, feature_set: str | None = None,
    data_dir: str | None = None, catalog: str | None = None,
) -> None:
    inference(model_path, out, config, test_path, features, feature_set, data_dir, catalog)

@task
def evaluate(preds: str, outdir: str, model_name: str) -> None:
    calc_metrics(preds, outdir, model_name)

@task
def render_summary_table(runs_dir: str, out: str, title: str) -> None:
    summary_table(runs_dir, out, title)

@task
def render_summary_plots(runs_dir: str, out: str, title: str) -> None:
    summary_plots(runs_dir, out, title)

@task
def fit_capsnet(config: str) -> None:
    train(config)

@task
def predict_capsnet_splits(config: str) -> None:
    inference(config_path=config)

@task
def evaluate_capsnet_split(preds: str, outdir: str, model_name: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(preds)
    adapted = out / "preds_metrics.parquet"
    df.rename(columns={"z_spec": "y_true", "z_pred": "y_pred"})[
        ["y_true", "y_pred"]
    ].to_parquet(adapted, index=False)
    calc_metrics(str(adapted), outdir, model_name)

def run_capsnet_pipeline(config: FlowConfig) -> None:
    train_cfg = config.train
    if train_cfg is None:
        raise ValueError("capsnet pipeline requires a train section")
    fit_capsnet(train_cfg["config"])
    predict_capsnet_splits(train_cfg["config"])
    nn_raw = yaml.safe_load(Path(train_cfg["config"]).read_text())
    for split in nn_raw.get("inference", {}).get("splits", []):
        out = str(split["out"])
        split_dir = str(Path(out).parent)
        evaluate_capsnet_split(out, split_dir, f"capsnet_{split['name']}")

def run_tabular_pipeline(config: FlowConfig) -> None:
    runs_root = config.runs_root
    prepared_root = cast(str, config.prepared_root)
    dataset = cast(dict[str, str], config.dataset)
    models = config.models
    summaries = config.summaries

    shutil.rmtree(prepared_root, ignore_errors=True)
    shutil.rmtree(runs_root, ignore_errors=True)

    split_tabular_data(
        dataset["input_csv"],
        dataset["data_config"],
        dataset["features"],
        prepared_root,
    )

    for model in models:
        config_path = model["config"]
        model_name = model.get("name", config_path.removeprefix("configs/").removesuffix(".yaml"))
        feature_set = model["feature_set"]
        outdir = model.get("outdir", f"{runs_root}/{model_name}_{feature_set}")

        fit_model(
            config_path,
            outdir,
            train_path=f"{prepared_root}/train.parquet",
            features=dataset["features"],
            feature_set=feature_set,
        )
        predict(
            f"{outdir}/model.pkl",
            f"{outdir}/preds.parquet",
            test_path=f"{prepared_root}/test.parquet",
            features=dataset["features"],
            feature_set=feature_set,
        )
        evaluate(f"{outdir}/preds.parquet", outdir, model_name)

    if summaries.get("enabled", True):
        title = str(summaries.get("title", "main"))
        render_summary_table(runs_root, f"{runs_root}/summary_table.html", title)
        render_summary_plots(runs_root, f"{runs_root}/summary_plots.html", title)

def run_nn_pipeline(config: FlowConfig) -> None:
    runs_root = config.runs_root
    models = config.models
    splits = config.splits
    summaries = config.summaries

    shutil.rmtree(runs_root, ignore_errors=True)

    for model in models:
        model_name = model["name"]
        config_path = model["config"]
        model_dir = model.get("train_outdir", f"{runs_root}/{model_name}")

        fit_model(config_path, model_dir)

        for split in splits:
            split_name = split["name"]
            split_dir = split.get("outdir", f"{runs_root}/runs_{split_name}/{model_name}")
            preds_path = f"{split_dir}/preds.parquet"

            predict(
                f"{model_dir}/best_model.pt",
                preds_path,
                config=config_path,
                data_dir=split["data_dir"],
                catalog=split["catalog"],
            )
            metric_suffix = split.get("metric_suffix", split_name)
            evaluate(preds_path, split_dir, f"{model_name}_{metric_suffix}")

    if summaries.get("enabled", False):
        for split in splits:
            split_name = split["name"]
            split_runs = split.get("summary_runs_dir", f"{runs_root}/runs_{split_name}")
            title = split.get("summary_title", f"nn_runs_{split_name}")
            render_summary_table(split_runs, f"{split_runs}/summary_table.html", title)
            render_summary_plots(split_runs, f"{split_runs}/summary_plots.html", title)

@flow(name="pipeline")
def pipeline(config_path: str) -> None:
    config = load_yaml_as(config_path, FlowConfig)
    pipeline_type = config.pipeline
    if pipeline_type == "tabular":
        run_tabular_pipeline(config)
    elif pipeline_type == "nn":
        run_nn_pipeline(config)
    elif pipeline_type == "capsnet":
        run_capsnet_pipeline(config)
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

@click.command()
@click.option("--config", "config_path", default="configs/flows/run_all.yaml", type=str)
def main(config_path: str) -> None:
    pipeline(config_path)

if __name__ == "__main__":
    main()
