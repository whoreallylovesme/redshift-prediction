from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run

from lib.config import EuclidFlowConfig, FlowConfig
from lib.inference import inference
from lib.tracking import Tracker, bind_run
from lib.utils.log import get_logger
from lib.utils.utils import load_yaml, load_yaml_as
from runtime.evaluate import calc_metrics
from runtime.render_plots import summary_plots
from runtime.render_table import summary_table
from runtime.split_tabular import prepare
from runtime.train import train

logger = get_logger()


@task(cache_policy=NO_CACHE)
def split_tabular_data(
    input_csv: str,
    data_config: str,
    features: str,
    outdir: str,
    id_column: str | None = None,
) -> None:
    prepare(input_csv, data_config, features, outdir, id_column=id_column)


@task(cache_policy=NO_CACHE)
def fit_model(
    config: str,
    outdir: str,
    train_path: str | None = None,
    features: str | None = None,
    feature_set: str | None = None,
    tracking_uri: str | None = None,
    run_id: str | None = None,
) -> None:
    with bind_run(tracking_uri, run_id):
        train(config, outdir, train_path, features, feature_set)


@task(cache_policy=NO_CACHE)
def predict(
    model_path: str,
    out: str,
    config: str | None = None,
    test_path: str | None = None,
    features: str | None = None,
    feature_set: str | None = None,
    data_dir: str | None = None,
    catalog: str | None = None,
    id_column: str | None = None,
) -> None:
    inference(
        model_path,
        out,
        config,
        test_path,
        features,
        feature_set,
        data_dir,
        catalog,
        id_column=id_column,
    )


@task(cache_policy=NO_CACHE)
def evaluate(preds: str, outdir: str, model_name: str) -> None:
    calc_metrics(preds, outdir, model_name)


@task(cache_policy=NO_CACHE)
def render_summary_table(runs_dir: str, out: str, title: str) -> None:
    summary_table(runs_dir, out, title)


@task(cache_policy=NO_CACHE)
def render_summary_plots(runs_dir: str, out: str, title: str) -> None:
    summary_plots(runs_dir, out, title)


@contextmanager
def tracked_method(
    uri: str | None,
    name: str,
    recipe: str,
    out: Path,
) -> Iterator[tuple[Tracker | None, str | None]]:
    if not uri:
        yield None, None
        return
    tracker = Tracker(uri, "photoz")
    run_id = tracker.start(name, {"recipe": recipe}, str(flow_run.id))
    success = False
    try:
        tracker.client.log_artifact(run_id, recipe, "recipe")
        yield tracker, run_id
        success = True
    finally:
        tracker.finish(run_id, success, out)


def track_scores(tracker: Tracker | None, run_id: str | None, split: str, output: Path) -> None:
    if tracker is not None and run_id is not None:
        import json

        metrics = json.loads((output / "report.json").read_text())["test_metrics"]
        tracker.log_metrics(run_id, split, {k: v for k, v in metrics.items() if pd.notna(v)})


@task(cache_policy=NO_CACHE)
def combined_report(
    output: Path,
    records: list[dict[str, Any]],
    title: str,
    run_ids: dict[str, str],
) -> None:
    from lib.report import render_report
    from photoz.report import PHOTOZ

    if not records:
        return
    report = render_report(
        output,
        records,
        PHOTOZ,
        title,
        {"prefect_flow_run": str(flow_run.id), "mlflow_runs": run_ids},
    )
    failures = [row["name"] for row in report["rows"] if row["status"] == "failed"]
    failures += [f"{p['a']} / {p['b']}" for p in report["pairs"] if p.get("status") == "failed"]
    if failures:
        raise ValueError(f"Report saved with evaluation failures: {', '.join(failures)}")


def fit_recipe(config: str, uri: str | None) -> None:
    recipe = Path(config)
    with tracked_method(uri, recipe.stem, config, recipe.parent) as (_, run_id):
        with bind_run(uri, run_id):
            train(config)


@task(cache_policy=NO_CACHE)
def fit_capsnet(config: str, tracking_uri: str | None = None, run_id: str | None = None) -> None:
    if run_id is None:
        fit_recipe(config, tracking_uri)
    else:
        with bind_run(tracking_uri, run_id):
            train(config)


@task(cache_policy=NO_CACHE)
def predict_capsnet_splits(config: str) -> None:
    inference(config_path=config)


@task(cache_policy=NO_CACHE)
def evaluate_capsnet_split(preds: str, outdir: str, model_name: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(preds)
    adapted = out / "preds_metrics.parquet"
    df.rename(columns={"z_spec": "y_true", "z_pred": "y_pred"})[["y_true", "y_pred"]].to_parquet(
        adapted, index=False
    )
    calc_metrics(str(adapted), outdir, model_name)


def run_capsnet_pipeline(config: FlowConfig, uri: str | None = None) -> None:
    train_cfg = config.train
    if train_cfg is None:
        raise ValueError("capsnet pipeline requires a train section")
    recipe = train_cfg["config"]
    nn_raw = yaml.safe_load(Path(recipe).read_text())
    root = Path(nn_raw["out_dir"])
    records: list[dict[str, Any]] = []
    with tracked_method(uri, Path(recipe).stem, recipe, root) as (tracker, run_id):
        fit_capsnet(recipe, uri, run_id)
        predict_capsnet_splits(recipe)
        inf = nn_raw["inference"]
        splits = inf["splits"] if "splits" in inf else [{"name": "valid", "out": inf["out"]}]
        for split in splits:
            out = Path(split["out"])
            evaluate_capsnet_split(str(out), str(out.parent), f"capsnet_{split['name']}")
            track_scores(tracker, run_id, split["name"], out.parent)
            records.append(
                {
                    "name": "CapsNet",
                    "split": split["name"],
                    "predictions": str(out),
                    "id_column": "objid",
                    "truth_column": "z_spec",
                    "prediction_column": "z_pred",
                    "details": {"recipe": recipe},
                }
            )
        combined_report(root, records, "CapsNet", {"CapsNet": run_id} if run_id else {})


def run_tabular_pipeline(config: FlowConfig, uri: str | None = None) -> None:
    runs_root = config.runs_root
    prepared_root = cast(str, config.prepared_root)
    dataset = cast(dict[str, str], config.dataset)
    summaries = config.summaries
    records: list[dict[str, Any]] = []
    run_ids: dict[str, str] = {}
    split_tabular_data(
        dataset["input_csv"],
        dataset["data_config"],
        dataset["features"],
        prepared_root,
        id_column=dataset.get("id_column"),
    )
    for model in config.models:
        config_path = model["config"]
        model_name = model.get("name", config_path.removeprefix("configs/").removesuffix(".yaml"))
        feature_set = model["feature_set"]
        label = f"{model_name}_{feature_set}"
        outdir = model.get("outdir", f"{runs_root}/{label}")
        out = Path(outdir)
        with tracked_method(uri, label, config_path, out) as (tracker, run_id):
            if run_id:
                run_ids[label] = run_id
            fit_model(
                config_path,
                outdir,
                train_path=f"{prepared_root}/train.parquet",
                features=dataset["features"],
                feature_set=feature_set,
                tracking_uri=uri,
                run_id=run_id,
            )
            preds = str(out / "preds.parquet")
            predict(
                str(out / "model.pkl"),
                preds,
                config=config_path,
                test_path=f"{prepared_root}/test.parquet",
                features=dataset["features"],
                feature_set=feature_set,
                id_column=dataset.get("id_column"),
            )
            evaluate(preds, outdir, model_name)
            track_scores(tracker, run_id, "test", out)
            records.append(
                {
                    "name": label,
                    "split": "test",
                    "predictions": preds,
                    "role": "held-out",
                    "id_column": dataset.get("id_column"),
                    "details": {"recipe": config_path, "feature_set": feature_set},
                }
            )
    title = str(summaries.get("title", "main"))
    combined_report(Path(runs_root), records, title, run_ids)
    if summaries.get("enabled", True):
        render_summary_table(runs_root, f"{runs_root}/summary_table.html", title)
        render_summary_plots(runs_root, f"{runs_root}/summary_plots.html", title)


def run_nn_pipeline(config: FlowConfig, uri: str | None = None) -> None:
    runs_root = config.runs_root
    splits = config.splits
    summaries = config.summaries
    records: list[dict[str, Any]] = []
    run_ids: dict[str, str] = {}
    for model in config.models:
        model_name = model["name"]
        config_path = model["config"]
        model_dir = model.get("train_outdir", f"{runs_root}/{model_name}")
        with tracked_method(uri, model_name, config_path, Path(model_dir)) as (tracker, run_id):
            if run_id:
                run_ids[model_name] = run_id
            fit_model(config_path, model_dir, tracking_uri=uri, run_id=run_id)
            for split in splits:
                split_name = split["name"]
                split_dir = split.get("outdir", f"{runs_root}/runs_{split_name}/{model_name}")
                preds_path = f"{split_dir}/preds.parquet"
                id_column = split.get("id_column", "object_id")
                predict(
                    f"{model_dir}/best_model.pt",
                    preds_path,
                    config=config_path,
                    data_dir=split["data_dir"],
                    catalog=split["catalog"],
                    id_column=id_column,
                )
                metric_suffix = split.get("metric_suffix", split_name)
                evaluate(preds_path, split_dir, f"{model_name}_{metric_suffix}")
                track_scores(tracker, run_id, split_name, Path(split_dir))
                records.append(
                    {
                        "name": model_name,
                        "split": split_name,
                        "predictions": preds_path,
                        "id_column": id_column,
                        "role": split.get("role", "unspecified"),
                        "details": {"recipe": config_path, "catalog": split["catalog"]},
                    }
                )
    combined_report(Path(runs_root), records, str(summaries.get("title", "photo-z")), run_ids)

    if summaries.get("enabled", False):
        for split in splits:
            split_name = split["name"]
            split_runs = split.get("summary_runs_dir", f"{runs_root}/runs_{split_name}")
            title = split.get("summary_title", f"nn_runs_{split_name}")
            render_summary_table(split_runs, f"{split_runs}/summary_table.html", title)
            render_summary_plots(split_runs, f"{split_runs}/summary_plots.html", title)


@task(cache_policy=NO_CACHE)
def fit_euclid_stage(config: str, tracking_uri: str | None = None) -> None:
    fit_recipe(config, tracking_uri)


@task(cache_policy=NO_CACHE)
def predict_euclid_stage(config: str) -> None:
    inference(config_path=config)


@task(cache_policy=NO_CACHE)
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


def run_euclid_pretrain(euclid: EuclidFlowConfig, runs_root: Path, uri: str | None = None) -> None:
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
            stage.config,
            runs_root / "configs" / f"clip_1m_{seed}.yaml",
            seed=seed,
            out_dir=str(out_dir),
        )
        fit_euclid_stage(config, uri)


def run_euclid_finetune(
    euclid: EuclidFlowConfig, runs_root: Path, tag: str, ft_dir: Path, uri: str | None = None
) -> None:
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
            stage.config,
            runs_root / "configs" / f"{tag}_{seed}.yaml",
            seed=seed,
            clip_state=str(clip_root / f"clip_1m_{clip_seed}" / STATE_NAME),
            out_dir=str(ft_dir),
        )
        fit_euclid_stage(config, uri)


def run_euclid_inference(
    euclid: EuclidFlowConfig,
    runs_root: Path,
    tag: str,
    ft_dir: Path,
) -> None:
    if not euclid.inference:
        return
    pred_dir = runs_root / "inference"
    seeds = [
        seed
        for _, seed in euclid.finetune.seed_pairs
        if not (pred_dir / f"{tag}_{seed}_pred.npz").exists()
    ]
    if not seeds:
        logger.info(f"[euclid] inference: every prediction in {pred_dir} exists, skipped")
        return
    config = derived_config(
        euclid.finetune.config,
        runs_root / "configs" / f"{tag}_inference.yaml",
        inference={"out_dir": str(pred_dir), "seeds": seeds, "state_dir": str(ft_dir)},
    )
    predict_euclid_stage(config)


def run_euclid_classical(
    euclid: EuclidFlowConfig,
    runs_root: Path,
    tag: str,
    ft_dir: Path,
    uri: str | None = None,
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
            stage.config,
            runs_root / "configs" / f"classical_{learner}.yaml",
            learner=learner,
            out_dir=str(out_dir),
            split_meta=str(split_meta),
        )
        fit_euclid_stage(config, uri)


def run_euclid_report(
    euclid: EuclidFlowConfig,
    runs_root: Path,
    tag: str,
    ft_dir: Path,
) -> None:
    stage = euclid.report
    if stage is None or not stage.enabled:
        return
    out_dir = runs_root / "report"
    if (out_dir / "facts.json").exists() and (out_dir / "report.html").exists():
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


def run_euclid_clip_pipeline(config: FlowConfig, uri: str | None = None) -> None:
    from lib.train.euclid_clip_finetune import EuclidClipConfig

    euclid = config.euclid
    if euclid is None:
        raise ValueError("euclid_clip pipeline requires a euclid section")
    if not euclid.finetune.seed_pairs:
        raise ValueError("euclid_clip pipeline requires fine-tune seed pairs")
    runs_root = Path(config.runs_root)
    tag = load_yaml_as(euclid.finetune.config, EuclidClipConfig).tag
    ft_dir = Path(euclid.finetune.out_dir) if euclid.finetune.out_dir else runs_root / tag
    run_euclid_pretrain(euclid, runs_root, uri)
    run_euclid_finetune(euclid, runs_root, tag, ft_dir, uri)
    run_euclid_inference(euclid, runs_root, tag, ft_dir)
    run_euclid_classical(euclid, runs_root, tag, ft_dir, uri)
    run_euclid_report(euclid, runs_root, tag, ft_dir)


@flow(name="pipeline")
def legacy_pipeline(
    config_path: str, tracking_uri: str | None = None, workers: int | None = None
) -> None:
    config = load_yaml_as(config_path, FlowConfig)
    if workers not in (None, 1):
        raise ValueError(
            "Historical flows use serial execution; configure workers in a methods flow"
        )
    tracking_uri = tracking_uri or config.tracking_uri
    pipeline_type = config.pipeline
    if pipeline_type == "tabular":
        run_tabular_pipeline(config, tracking_uri)
    elif pipeline_type == "nn":
        run_nn_pipeline(config, tracking_uri)
    elif pipeline_type == "capsnet":
        run_capsnet_pipeline(config, tracking_uri)
    elif pipeline_type == "euclid_clip":
        run_euclid_clip_pipeline(config, tracking_uri)
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
