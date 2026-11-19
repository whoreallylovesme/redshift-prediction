from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from tempfile import gettempdir
from typing import Any, cast
from uuid import UUID

import joblib
from filelock import FileLock
from hydra.utils import instantiate
from prefect import flow, task, unmapped
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import INPUTS, NO_CACHE, TASK_SOURCE
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterId
from prefect.client.schemas.objects import TaskRun
from prefect.client.schemas.sorting import TaskRunSort
from prefect.futures import PrefectFuture
from prefect.runtime import flow_run
from prefect.task_runners import TaskRunner, ThreadPoolTaskRunner

from lib.config import PytorchTrainConfig, SklearnTrainConfig
from lib.config.experiment import (
    Dataset,
    Experiment,
    External,
    ImageSplit,
    Lightning,
    MethodConfig,
    PreparationStep,
    Saved,
    Tabular,
    image_splits,
)
from lib.config.snapshot import (
    create_snapshot,
    public_config,
    recording_configs,
    write_method_recipes,
)
from lib.report.task import metric_suite, scores
from lib.tracking.mlflow import Tracker, bind_run, finish_with_snapshots
from lib.train.sklearn import fit_regressor
from lib.utils.artifacts import (
    input_fingerprints,
    sha256_path,
    verify_artifact_manifest,
    write_artifact_manifest,
)
from lib.utils.utils import load_yaml_as
from lib.workflow.data import code_version, digest, prepare_dataset, symbol, table

_ACCELERATOR = Path(gettempdir()) / "astro-lightning"
_FIT_CACHE = TASK_SOURCE + (INPUTS - "data" - "recipe")
_PREPARATION_CACHE = TASK_SOURCE + (INPUTS - "snapshot")


def accelerator_lock() -> FileLock:
    import getpass

    return FileLock(str(_ACCELERATOR) + "-" + getpass.getuser() + ".lock")


@task(cache_policy=NO_CACHE)
def prepare(cfg: Dataset, out: Path, snapshot: Path | None = None) -> dict[str, Any]:
    with recording_configs(snapshot):
        return prepare_dataset(cfg, out)


@task(cache_policy=NO_CACHE)
def prepare_resource(
    step: PreparationStep,
    fingerprints: dict[str, str],
    version: str,
    snapshot: Path | None = None,
) -> dict[str, str]:
    with recording_configs(snapshot):
        kwargs = instantiate(step.params)
        if not isinstance(kwargs, Mapping):
            raise TypeError("Preparation params must instantiate to a mapping")
        symbol(step.action)(**kwargs)
    return {str(path): sha256_path(path) for path in step.outputs}


def run_preparation(
    config: Experiment, version: str, snapshot: Path
) -> dict[str, dict[str, str]]:
    pending = set(config.preparation)
    futures: dict[str, PrefectFuture[Any]] = {}
    results: dict[str, dict[str, str]] = {}
    while pending:
        ready = [
            name
            for name in config.preparation
            if name in pending
            and all(dependency in results for dependency in config.preparation[name].needs)
        ]
        if not ready:
            raise ValueError("Preparation dependency graph cannot make progress")
        for name in ready:
            step = config.preparation[name]
            cached = config.cache and step.cache
            futures[name] = prepare_resource.with_options(
                task_run_name=f"prepare-{name}",
                cache_policy=_PREPARATION_CACHE if cached else NO_CACHE,
                persist_result=cached,
            ).submit(step, input_fingerprints(step.inputs), version, snapshot)
            pending.remove(name)
        for name in ready:
            results[name] = futures[name].result()
    return {name: results[name] for name in config.preparation}


@task(cache_policy=NO_CACHE)
def fit(
    data: dict[str, Any],
    recipe: Path,
    task_name: str,
    threads: int,
    cache_inputs: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    cfg = load_yaml_as(str(recipe), SklearnTrainConfig)
    train = table(Path(data["train"]))
    suite = metric_suite(task_name)
    model, summary = fit_regressor(
        train,
        data["features"],
        data["target"],
        cfg,
        threads=threads,
        weight_column=data.get("weight_column"),
        metrics=lambda y, pred: scores(suite, y.to_numpy(), pred),
    )
    return model, summary


@task(cache_policy=NO_CACHE)
def save_tabular_model(fitted: tuple[Any, dict[str, Any]], out: Path) -> Path:
    model, summary = fitted
    out.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, out / "model.pkl")
    (out / "train_report.json").write_text(json.dumps(summary, indent=2))
    return out / "model.pkl"


@task(cache_policy=NO_CACHE)
def predict_table(
    model_path: Path, source: str, features: list[str], target: str, path: Path
) -> str:
    model = joblib.load(model_path)
    held = table(Path(source))
    prediction = held.copy()
    if target in held:
        prediction["y_true"] = held[target]
    prediction["y_pred"] = model.predict(held[features])
    prediction.to_parquet(path, index=False)
    return str(path)


@task(cache_policy=NO_CACHE)
def external_prediction(
    cfg: External, recipe: Path, out: Path, uri: str | None, run_id: str | None,
    snapshot: Path | None = None,
) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = accelerator_lock() if cfg.resource == "accelerator" else nullcontext()
    with lock, bind_run(uri, run_id), recording_configs(snapshot):
        return str(symbol(cfg.model)(config_path=recipe, out_dir=out, **cfg.options))


@task(cache_policy=NO_CACHE)
def image_fit(recipe: Path, out: Path, uri: str | None, run_id: str | None) -> str:
    from lib.train.regression_lightning import fit_images

    with accelerator_lock(), bind_run(uri, run_id):
        fit_images(load_yaml_as(recipe, PytorchTrainConfig), out)
    weights = out / "best_model.pt"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return str(weights)


@task(cache_policy=NO_CACHE)
def image_predict(
    recipe: Path, weights: str, split: str, dataset: ImageSplit, out: Path, id_column: str
) -> str:
    from lib.inference.images import predict_catalog

    output = out / f"predictions_{split}.parquet"
    with accelerator_lock():
        recipe_cfg = load_yaml_as(recipe, PytorchTrainConfig)
        predict_catalog(
            recipe_cfg, recipe_cfg.image_catalog, Path(weights), dataset.images, dataset.catalog,
            output, id_column,
            held_out=True,
        )
    return str(output)


@task(cache_policy=NO_CACHE)
def evaluate_and_compare(
    records: list[dict[str, Any]],
    task_name: str,
    name: str,
    repeats: int,
    code_sha256: str,
    sources: dict[str, str],
) -> dict[str, Any]:
    from lib.report import build_report_data

    return build_report_data(
        records, metric_suite(task_name), name, {"bootstrap_replicates": repeats}
    )


@task(cache_policy=NO_CACHE)
def publish_report(out: Path, report: dict[str, Any]) -> None:
    from lib.report import write_report

    write_report(out, report)


@flow(name="rebuild-report")
def rebuild_report(directory: Path) -> None:
    publish_report(directory, json.loads((directory / "report-data.json").read_text()))


def submit_methods(
    config: Experiment,
    directory: Path,
    data: dict[str, Any],
    version: str,
    run_ids: dict[str, str],
    recipes: dict[str, Path],
    dataset_error: Exception | None = None,
) -> list[tuple[str, str, str | PrefectFuture[Any] | Exception]]:
    submitted: list[tuple[str, str, str | PrefectFuture[Any] | Exception]] = []
    for name, method in config.methods.items():
        out = directory / "methods" / name
        if dataset_error is not None and isinstance(method, (Tabular, Lightning)):
            labels = list(config.dataset.evaluation_files) if config.dataset else []
            submitted.extend((name, label, dataset_error) for label in labels or [method.split])
            continue
        if isinstance(method, Saved):
            submitted.append((name, method.split, str(method.predictions)))
        elif isinstance(method, External):
            prediction = external_prediction.with_options(task_run_name=f"{name}-predict").submit(
                method,
                recipes[name],
                out,
                config.tracking_uri,
                run_ids.get(name),
                snapshot=directory / "config_snapshot",
            )
            submitted.append((name, method.split, prediction))
        elif isinstance(method, Tabular):
            # Run directories change on every execution. Hash their contents and training schema.
            cache_inputs = None
            if config.cache:
                keys = (
                    "train_sha256",
                    "features",
                    "missing_features",
                    "target",
                    "weight_column",
                )
                cache_inputs = {key: data.get(key) for key in keys}
                cache_inputs.update(
                    code=version,
                    recipe=digest(recipes[name]),
                )
            fitted = fit.with_options(
                task_run_name=f"{name}-fit",
                cache_policy=_FIT_CACHE if config.cache else NO_CACHE,
                persist_result=config.cache,
            ).submit(data, recipes[name], config.task, config.threads_per_method, cache_inputs)
            model_path = save_tabular_model.with_options(task_run_name=f"{name}-save").submit(
                fitted, out, return_state=False
            )
            splits = data["evaluation_files"] or {method.split: data["test"]}
            for split, source in splits.items():
                path = out / (
                    f"predictions_{split}.parquet"
                    if data["evaluation_files"]
                    else "predictions.parquet"
                )
                pred = predict_table.with_options(task_run_name=f"{name}-{split}-predict").submit(
                    model_path,
                    source,
                    data["features"],
                    data["target"],
                    path,
                    return_state=False,
                )
                submitted.append((name, split, pred))
        elif isinstance(method, Lightning):
            assert config.dataset is not None and config.dataset.id_column is not None
            weights = image_fit.with_options(task_run_name=f"{name}-fit").submit(
                recipes[name], out, config.tracking_uri, run_ids.get(name)
            )
            images = image_splits(config.dataset, method.split)
            catalogs = data["evaluation_files"] or {method.split: data["test"]}
            splits = list(catalogs)
            datasets = [
                ImageSplit(images=images[label], catalog=catalogs[label]) for label in splits
            ]
            predictions = image_predict.with_options(task_run_name=f"{name}-{{split}}").map(
                unmapped(recipes[name]), unmapped(weights), splits, datasets, unmapped(out),
                unmapped(config.dataset.id_column),
            )
            submitted.extend(
                (name, split, prediction)
                for split, prediction in zip(splits, predictions, strict=True)
            )
    return submitted


def training_history(path: Path, directory: Path) -> Path | None:
    per_prediction = path.with_suffix(".history.json")
    if per_prediction.is_file():
        return per_prediction
    for name in ("history.json", "history.csv", "history.jsonl"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    log = directory / "log.jsonl"
    if log.is_file():
        with log.open() as stream:
            first = next((line for line in stream if line.strip()), None)
        if first is not None:
            try:
                entry = json.loads(first)
            except ValueError:
                entry = None
            if isinstance(entry, dict) and {"epoch", "step"} & entry.keys():
                return log
    summary = directory / "train_report.json"
    if summary.is_file():
        content = json.loads(summary.read_text())
        if isinstance(content, dict) and isinstance(content.get("history"), list):
            return summary
    named = list(directory.glob("*_history.json"))
    return named[0] if len(named) == 1 else None


def report_record(
    name: str,
    split: str,
    method: MethodConfig,
    prediction: str | PrefectFuture[Any],
    id_column: str | None,
    directory: Path | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = method.report.model_dump(mode="json", exclude_unset=True)
    record.update(name=name, split=split)
    if isinstance(method, Lightning):
        record.setdefault("role", "held-out")
    if id_column and not isinstance(method, Saved):
        record.setdefault("id_column", id_column)
    path = prediction if isinstance(prediction, str) else prediction.result()
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    if isinstance(method, External) and directory:
        provenance = directory / "provenance.json"
        if provenance.is_file():
            record["details"] = json.loads(provenance.read_text())
    if isinstance(method, External) and method.report_from_predictor:
        from lib.report.posterior_metadata import PosteriorMetadata

        source = Path(path).resolve()
        if directory is None or not source.is_relative_to(directory.resolve()):
            raise ValueError("Predictor metadata must belong to this method's run directory")
        posterior = PosteriorMetadata.read(source)
        for key, value in posterior.model_dump(mode="json").items():
            if key in record and record[key] != value:
                raise ValueError(f"Configured {key} disagrees with predictor metadata")
            record[key] = value
    if isinstance(method, Tabular):
        summary = Path(path).with_name("train_report.json")
        record["details"] = json.loads(summary.read_text())
    if isinstance(method, (Lightning, External)) and "history" not in record:
        history = training_history(Path(path), directory or Path(path).parent)
        if history is not None:
            record["history"] = str(history)
    record.update(predictions=str(path), status="completed")
    return record


def stage_history(flow_id: str) -> list[dict[str, Any]]:
    with get_client(sync_client=True) as client:
        query = FlowRunFilter(id=FlowRunFilterId(any_=[UUID(flow_id)]))
        stages: list[TaskRun] = []
        while page := client.read_task_runs(
            flow_run_filter=query, sort=TaskRunSort.ID_DESC, offset=len(stages)
        ):
            stages.extend(page)
    return [
        {
            "stage": stage.name,
            "state": stage.state_name,
            "seconds": stage.total_run_time.total_seconds(),
        }
        for stage in sorted(stages, key=lambda item: str(item.created))
    ]


def finalize_artifacts(
    config: Experiment,
    directory: Path,
    data: dict[str, Any],
    version: str,
    records: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, method in config.methods.items():
        if isinstance(method, Saved):
            continue
        rows = [
            row
            for row in records
            if row["name"] == name and row["status"] == "completed"
        ]
        if not rows:
            continue
        out = directory / "methods" / name
        rows = [row for row in rows if Path(row["predictions"]).is_relative_to(out)]
        if not rows:
            continue
        names = {
            "tabular": ("model.pkl", "train_report.json"),
            "lightning": (
                "best_model.pt",
                "last_model.pt",
                "image_normalization.npz",
                "history.json",
            ),
            "external": ("provenance.json",),
        }
        paths = [out / item for item in names[method.kind]]
        paths.extend(Path(row["predictions"]) for row in rows)
        for row in rows:
            history = row.get("history")
            if history and Path(history).is_relative_to(out):
                paths.append(Path(history))
        write_artifact_manifest(
            out,
            [path for path in paths if path.is_file()],
            method=name,
            kind=method.kind,
            estimator=method.model,
            task=config.task,
            code_sha256=version,
            dataset_sha256=data.get("source_sha256"),
            features=data.get("features", []),
            id_column=data.get("id_column"),
            target=data.get("target"),
            config=public_config(method.model_dump(mode="json")),
        )
        verify_artifact_manifest(out)
        manifest = out / "artifact.json"
        result[name] = {
            "path": manifest.relative_to(directory).as_posix(),
            "sha256": digest(manifest),
        }
    return result


@flow(name="regression-experiment", flow_run_name="{config.name}")
def experiment(config: Experiment) -> Path:
    directory = config.output.resolve() / str(flow_run.id)
    if flow_run.run_count > 1:
        directory /= f"attempt-{flow_run.run_count}"
    directory.mkdir(parents=True, exist_ok=True)
    configuration = create_snapshot(directory / "config_snapshot", config)
    snapshot = public_config(config.model_dump(mode="json"))
    (directory / "experiment.json").write_text(json.dumps(snapshot, indent=2))
    plugins = [
        config.task,
        *(m.model for m in config.methods.values() if isinstance(m, External)),
        *(step.action for step in config.preparation.values()),
    ]
    if config.dataset:
        plugins.extend(x for x in (config.dataset.reader, config.dataset.cleaner) if x)
    version = code_version(plugins)
    data: dict[str, Any] = {}
    dataset_error: Exception | None = None
    run_ids: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    tracker = Tracker(config.tracking_uri, config.name) if config.tracking_uri else None
    finished = False
    failures: list[str] = []
    preparation_results: dict[str, dict[str, str]] = {}
    try:
        preparation_results = run_preparation(config, version, configuration)
        uses_catalog = any(
            isinstance(m, (Tabular, Lightning))
            for m in config.methods.values()
        )
        if config.dataset is not None and (uses_catalog or config.dataset.features):
            try:
                data = prepare(config.dataset, directory / "data", snapshot=configuration)
            except Exception as exc:
                dataset_error = ValueError(f"Dataset preparation failed: {exc}")
        recipes = write_method_recipes(config, configuration, data)
        if tracker:
            for name, method in config.methods.items():
                params: dict[str, Any] = {
                    "method": method.kind,
                    "task": config.task,
                    "code_sha256": version,
                }
                if name in recipes:
                    params["recipe_sha256"] = digest(recipes[name])
                if data:
                    params["dataset_sha256"] = data["source_sha256"]
                run_ids[name] = tracker.start(name, params, str(flow_run.id))
        (directory / "tracking.json").write_text(json.dumps(run_ids, indent=2))
        pending = submit_methods(config, directory, data, version, run_ids, recipes, dataset_error)
        for name, split, method_result in pending:
            try:
                if isinstance(method_result, Exception):
                    raise method_result
                row = report_record(
                    name,
                    split,
                    config.methods[name],
                    method_result,
                    data.get("id_column") or (config.dataset.id_column if config.dataset else None),
                    directory / "methods" / name,
                )
            except Exception as exc:
                row = {"name": name, "split": split, "status": "failed", "reason": str(exc)}
            records.append(row)
        artifact_manifests = finalize_artifacts(config, directory, data, version, records)
        sources: dict[str, str] = {}
        for row in records:
            for field in ("predictions", "reference", "history"):
                path = row.get(field)
                if isinstance(path, str) and path not in sources and Path(path).is_file():
                    sources[path] = digest(Path(path))
        # Thumbnail paths inside prediction tables can change contents independently of the table.
        cache_report = config.cache and not any(row.get("thumbnail_column") for row in records)
        report = evaluate_and_compare.with_options(
            cache_policy=TASK_SOURCE + INPUTS if cache_report else NO_CACHE,
            persist_result=cache_report,
        )(records, config.task, config.name, config.bootstrap_replicates, version, sources)
        report["metadata"] = {
            "prefect_flow_run": str(flow_run.id),
            "mlflow_runs": run_ids,
            "code_sha256": version,
            "configuration": snapshot,
            "preparation": preparation_results,
            "artifacts": artifact_manifests,
            "data": data if dataset_error is None else {"error": str(dataset_error)},
            "sources": sources,
        }
        report["metadata"]["stages"] = stage_history(str(flow_run.id))
        publish_report(directory, report)
        create_table_artifact(
            key="regression-metrics",
            table=[
                {"method": r["name"], "split": r["split"], "status": r["status"], **r["metrics"]}
                for r in report["rows"]
            ],
            description=f"{config.name} · {directory / 'report.html'}",
        )
        if tracker:
            for method_name, run_id in run_ids.items():
                method_cfg = config.methods[method_name]
                settings = public_config(method_cfg.model_dump(mode="json"))
                tracker.client.log_dict(run_id, settings, "method.json")
                if method_name in recipes:
                    tracker.client.log_artifact(run_id, str(recipes[method_name]), "recipe")
                summary = directory / "methods" / method_name / "train_report.json"
                if isinstance(method_cfg, Tabular) and summary.is_file():
                    fitted = json.loads(summary.read_text())
                    tracker.client.log_dict(run_id, fitted, "fit.json")
                    tracker.log_metrics(run_id, "fit_validation", fitted.get("val_metrics", {}))
                    for parameter, value in fitted.get("best_params", {}).items():
                        tracker.client.log_param(run_id, f"best.{parameter}", value)
                tracker.client.log_artifact(run_id, str(directory / "metrics.csv"), "evaluation")
                manifest = directory / "methods" / method_name / "artifact.json"
                if manifest.is_file():
                    tracker.client.log_artifact(run_id, str(manifest), "evaluation")
            for row in report["rows"]:
                tracker.log_metrics(run_ids[row["name"]], row["split"], row["metrics"])
        failures = [r["name"] for r in report["rows"] if r["status"] == "failed"]
        if dataset_error and not uses_catalog:
            failures.append("dataset preparation")
        failures.extend(
            f"{p['a']} / {p['b']}" for p in report["pairs"] if p.get("status") == "failed"
        )
        finished = True
    finally:
        finish_with_snapshots(
            tracker,
            configuration,
            directory,
            flow_run.run_count,
            {run_id: finished and name not in failures for name, run_id in run_ids.items()},
        )
    if failures:
        raise RuntimeError(f"Failed methods: {', '.join(failures)}; report: {directory}")
    return directory


def run(config: Experiment) -> Path:
    return experiment.with_options(
        task_runner=cast(
            TaskRunner[PrefectFuture[Any]], ThreadPoolTaskRunner(max_workers=config.workers)
        ),
    )(config)
