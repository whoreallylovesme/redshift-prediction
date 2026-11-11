import shutil
from pathlib import Path
from typing import cast

import click
import pandas as pd
import yaml
from prefect import flow, task

from lib.config import EuclidFlowConfig, FlowConfig
from lib.inference import inference
from lib.utils.log import get_logger
from lib.utils.utils import load_yaml, load_yaml_as
from runtime.evaluate import calc_metrics
from runtime.render_plots import summary_plots
from runtime.render_table import summary_table
from runtime.split_tabular import prepare
from runtime.train import train

logger = get_logger()


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

@task
def fit_euclid_stage(config: str) -> None:
    train(config)

@task
def predict_euclid_stage(config: str) -> None:
    inference(config_path=config)

@task
def report_euclid_stage(config: str) -> None:
    from runtime.euclid_report import build_report

    build_report(Path(config))

def derived_config(base: str, out: Path, **overrides: object) -> str:
    raw = load_yaml(base)
    if not isinstance(raw, dict):
        raise ValueError(f"{base} must be a YAML mapping")
    raw.update(overrides)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return str(out)

def run_euclid_pretrain(euclid: EuclidFlowConfig, runs_root: Path) -> None:
    from lib.train.euclid_clip_pretrain import STATE_NAME

    stage = euclid.pretrain
    if stage is None or not stage.enabled:
        return
    seeds = stage.seeds or sorted({clip_seed for clip_seed, _ in euclid.finetune.seed_pairs})
    for seed in seeds:
        out_dir = runs_root / f"clip_1m_{seed}"
        if (out_dir / STATE_NAME).exists():
            logger.info(f"[euclid] pretrain seed {seed}: {out_dir / STATE_NAME} exists, skipped")
            continue
        config = derived_config(
            stage.config, runs_root / "configs" / f"clip_1m_{seed}.yaml",
            seed=seed, out_dir=str(out_dir),
        )
        fit_euclid_stage(config)

def run_euclid_finetune(euclid: EuclidFlowConfig, runs_root: Path, tag: str, ft_dir: Path) -> None:
    from lib.train.euclid_clip_pretrain import STATE_NAME

    stage = euclid.finetune
    if not stage.enabled:
        return
    clip_root = Path(stage.clip_root) if stage.clip_root else runs_root
    for clip_seed, seed in stage.seed_pairs:
        meta = ft_dir / f"{tag}_{seed}.json"
        if meta.exists():
            logger.info(f"[euclid] fine-tune seed {seed}: {meta} exists, skipped")
            continue
        config = derived_config(
            stage.config, runs_root / "configs" / f"{tag}_{seed}.yaml",
            seed=seed, clip_state=str(clip_root / f"clip_1m_{clip_seed}" / STATE_NAME),
            out_dir=str(ft_dir),
        )
        fit_euclid_stage(config)

def run_euclid_inference(
    euclid: EuclidFlowConfig, runs_root: Path, tag: str, ft_dir: Path,
) -> None:
    if not euclid.inference:
        return
    pred_dir = runs_root / "inference"
    seeds = [
        seed for _, seed in euclid.finetune.seed_pairs
        if not (pred_dir / f"{tag}_{seed}_pred.npz").exists()
    ]
    if not seeds:
        logger.info(f"[euclid] inference: every prediction in {pred_dir} exists, skipped")
        return
    config = derived_config(
        euclid.finetune.config, runs_root / "configs" / f"{tag}_inference.yaml",
        inference={"out_dir": str(pred_dir), "seeds": seeds, "state_dir": str(ft_dir)},
    )
    predict_euclid_stage(config)

def run_euclid_classical(
    euclid: EuclidFlowConfig, runs_root: Path, tag: str, ft_dir: Path,
) -> None:
    stage = euclid.classical
    if stage is None or not stage.enabled:
        return
    split_meta = ft_dir / f"{tag}_{euclid.finetune.seed_pairs[0][1]}.json"
    for learner in stage.learners:
        out_dir = runs_root / "classical" / learner
        if (out_dir / "complete.json").exists():
            logger.info(f"[euclid] classical {learner}: {out_dir} is complete, skipped")
            continue
        config = derived_config(
            stage.config, runs_root / "configs" / f"classical_{learner}.yaml",
            learner=learner, out_dir=str(out_dir), split_meta=str(split_meta),
        )
        fit_euclid_stage(config)

def run_euclid_report(
    euclid: EuclidFlowConfig, runs_root: Path, tag: str, ft_dir: Path,
) -> None:
    stage = euclid.report
    if stage is None or not stage.enabled:
        return
    out_dir = runs_root / "report"
    if (out_dir / "facts.json").exists():
        logger.info(f"[euclid] report: {out_dir / 'facts.json'} exists, skipped")
        return
    pairs = euclid.finetune.seed_pairs
    clip_root = Path(euclid.finetune.clip_root) if euclid.finetune.clip_root else runs_root
    overrides: dict[str, object] = {
        "tag": tag,
        "seeds": [seed for _, seed in pairs],
        "meta_dir": str(ft_dir),
        "pred_dir": str(runs_root / "inference" if euclid.inference else ft_dir),
        "clip_dir": str(clip_root),
        "clip_seeds": sorted({clip_seed for clip_seed, _ in pairs}),
        "out_dir": str(out_dir),
    }
    if euclid.classical is not None:
        overrides["classical_dir"] = str(runs_root / "classical")
        overrides["learners"] = euclid.classical.learners
    config = derived_config(stage.config, runs_root / "configs" / "report.yaml", **overrides)
    report_euclid_stage(config)

def run_euclid_clip_pipeline(config: FlowConfig) -> None:
    from lib.train.euclid_clip_finetune import EuclidClipConfig

    euclid = config.euclid
    if euclid is None:
        raise ValueError("euclid_clip pipeline requires a euclid section")
    if not euclid.finetune.seed_pairs:
        raise ValueError("euclid_clip pipeline requires fine-tune seed pairs")
    runs_root = Path(config.runs_root)
    tag = load_yaml_as(euclid.finetune.config, EuclidClipConfig).tag
    ft_dir = Path(euclid.finetune.out_dir) if euclid.finetune.out_dir else runs_root / tag
    run_euclid_pretrain(euclid, runs_root)
    run_euclid_finetune(euclid, runs_root, tag, ft_dir)
    run_euclid_inference(euclid, runs_root, tag, ft_dir)
    run_euclid_classical(euclid, runs_root, tag, ft_dir)
    run_euclid_report(euclid, runs_root, tag, ft_dir)

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
    elif pipeline_type == "euclid_clip":
        run_euclid_clip_pipeline(config)
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

@click.command()
@click.option("--config", "config_path", default="configs/flows/run_all.yaml", type=str)
def main(config_path: str) -> None:
    pipeline(config_path)

if __name__ == "__main__":
    main()
