from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from lib.dataloaders.dataset import exact_index, matching_id_types

from .examples import example_assets, native_posteriors, posterior_diagnostics
from .task import Floats, ReportTask, resolve_task, scores

Record = dict[str, Any]
POINT_LIMIT = 600


def read_frame(path: str | Path | pd.DataFrame, id_column: str | None = None) -> pd.DataFrame:
    if isinstance(path, pd.DataFrame):
        return path
    path = Path(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(
            path,
            sep="\t" if path.suffix.lower() == ".tsv" else ",",
            dtype={id_column: "string"} if id_column else None,
        )
    return pd.read_parquet(path)


def identity(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    exact_index(frame[column], column)
    return frame.set_index(column, drop=False)


def numeric(series: pd.Series) -> Floats:
    return cast(
        Floats, pd.to_numeric(series, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    )


def boolean(series: pd.Series, column: str) -> pd.Series:
    if not series.dropna().isin([True, False, 0, 1]).all():
        raise ValueError(f"{column} must contain booleans or 0/1")
    return series.eq(True).fillna(False)


def prepare_frame(method: Record, task: ReportTask) -> tuple[pd.DataFrame, bool]:
    id_col = method.get("id_column", "object_id")
    predicted = read_frame(method["predictions"], id_col)
    if "id_column" not in method:
        id_col = predicted.attrs.get(
            "prediction_id_column", id_col if id_col in predicted else None
        )
    if id_col is not None:
        predicted = identity(predicted, id_col)
    truth_col = method.get("truth_column", "y_true")
    pred_col = method.get("prediction_column", "y_pred")
    reference = method.get("reference")
    if reference is not None:
        if id_col is None:
            raise ValueError("Matching a reference requires an explicit prediction ID")
        ref_id = method.get("reference_id_column") or id_col
        ref = identity(read_frame(reference, ref_id), ref_id)
        if not matching_id_types(predicted.index, ref.index):
            raise ValueError("Prediction and reference identifiers have different types")
        unknown = predicted.index.difference(ref.index)
        if len(unknown):
            raise ValueError(f"{len(unknown)} prediction IDs outside the reference")
        ref_truth_col = method.get("reference_truth_column") or truth_col
        ref_truth = numeric(ref[ref_truth_col])
        if truth_col and truth_col in predicted:
            native_truth = numeric(predicted[truth_col])
            expected = pd.Series(ref_truth, index=ref.index).loc[predicted.index].to_numpy()
            supplied = np.isfinite(native_truth)
            if not np.allclose(native_truth[supplied], expected[supplied], rtol=1e-6, atol=1e-6):
                raise ValueError("Prediction labels disagree with the reference")
        frame = ref.copy()
        model_columns = {pred_col} | {
            method[key]
            for key in ("posterior_column", "upper_column", "lower_column")
            if method.get(key)
        }
        missing = model_columns.difference(predicted.columns)
        if missing:
            raise ValueError(f"Missing prediction columns: {sorted(missing)}")
        for col in predicted:
            if col not in frame or col in model_columns:
                frame[col] = predicted[col].reindex(ref.index)
        represented = ref.index.isin(predicted.index)
        truth = ref_truth
    else:
        frame = predicted.copy()
        represented = np.ones(len(frame), dtype=bool)
        if truth_col and truth_col not in frame and "truth_column" in method:
            raise ValueError(f"Missing configured truth column {truth_col!r}")
        truth = numeric(frame[truth_col]) if truth_col in frame else np.full(len(frame), np.nan)
    if "image_split_overlap" in predicted.attrs:
        frame.attrs["image_split_overlap"] = predicted.attrs["image_split_overlap"]
    frame["_truth"] = truth
    frame["_pred"] = numeric(predicted[pred_col].reindex(frame.index))
    status_col = method.get("status_column")
    if not status_col and "status" in predicted.columns:
        status_col = "status"
    status = (
        predicted[status_col].reindex(frame.index).astype("string").fillna("missing_status")
        if status_col
        else pd.Series("ok", index=frame.index, dtype="string")
    )
    configured = method.get("ok_statuses")
    allowed = {
        str(v)
        for v in (
            configured if configured is not None else ["ok", "complete", "completed", "success"]
        )
    }
    admission_col = method.get("admitted_column")
    admitted = boolean(frame[admission_col], admission_col) if admission_col else represented
    frame["_admitted"] = admitted
    frame["_predicted"] = represented & admitted & status.isin(allowed) & np.isfinite(frame._pred)
    valid_truth = task.valid_target(truth)
    frame["_evaluated"] = frame._predicted & valid_truth
    reason = pd.Series("", index=frame.index, dtype="string")
    reason[~valid_truth] = "invalid or missing reference"
    reason[~np.isfinite(frame._pred)] = "nonfinite prediction"
    reason[~status.isin(allowed)] = status[~status.isin(allowed)]
    reason[~represented] = "missing prediction row"
    if admission_col:
        reason[~admitted] = "ineligible"
    frame["_reason"] = reason
    group_col = method.get("group_column")
    if group_col:
        frame["_group"] = frame[group_col]
    return frame, id_col is not None


def slice_frames(frame: pd.DataFrame, method: Record) -> dict[str, pd.DataFrame]:
    subsets = {"All objects": frame}
    for col in method.get("slices", []):
        if col not in frame:
            continue
        labels = frame[col].astype("string").fillna("(missing)")
        if labels.nunique() > 64:
            raise ValueError(f"Slice {col!r} has more than 64 levels; configure bins")
        for value, rows in frame.groupby(labels, sort=True):
            subsets[f"{col} = {value}"] = rows
    for col, edges in method.get("slice_bins", {}).items():
        if col not in frame:
            continue
        bins = pd.cut(numeric(frame[col]), edges, include_lowest=True)
        labels = pd.Series(bins, index=frame.index).astype("string").fillna("(outside or missing)")
        for value, rows in frame.groupby(labels, sort=True):
            subsets[f"{col} ∈ {value}"] = rows
    return subsets


def bin_edges(values: Floats, bins: int) -> Floats:
    lo, hi = float(values.min()), float(values.max())
    edges = np.unique(np.linspace(lo, hi, bins + 1))
    if len(edges) == 1:
        width = max(abs(lo) * 1e-6, 0.5)
        return np.array([lo - width, hi + width])
    return edges


def histogram(values: Floats) -> Record:
    if not len(values):
        return {"x": [], "y": [], "width": []}
    counts, edges = np.histogram(
        values, bins=bin_edges(values, min(40, max(4, int(np.sqrt(len(values))))))
    )
    return {
        "x": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "y": counts.tolist(),
        "width": np.diff(edges).tolist(),
    }


def coverage(frame: pd.DataFrame) -> Record:
    return {
        "requested": len(frame),
        "admitted": int(frame._admitted.sum()),
        "predicted": int(frame._predicted.sum()),
        "evaluated": int(frame._evaluated.sum()),
        "reasons": dict(Counter(frame.loc[~frame._evaluated, "_reason"].tolist())),
    }


def diagnostics(frame: pd.DataFrame, task: ReportTask, method: Record, has_id: bool) -> Record:
    known = frame.loc[task.valid_target(numeric(frame._truth))]
    scored = frame.loc[frame._evaluated]
    t, p = numeric(scored._truth), numeric(scored._pred)
    sample = np.linspace(0, len(t) - 1, min(POINT_LIMIT, len(t)), dtype=int)
    chart: Record = {
        "sample": {"x": t[sample].tolist(), "y": p[sample].tolist()},
        "target": histogram(numeric(known._truth)),
        "residual": histogram(task.residual(t, p)),
        "bins": [],
        "guides": {},
        "interval": None,
    }
    if len(t):
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        x = np.array([lo, hi]) if lo < hi else np.array([lo - 0.5, hi + 0.5])
        chart["guides"] = {k: {"x": x.tolist(), "y": y.tolist()} for k, y in task.guides(x).items()}
        if len(t) > POINT_LIMIT:
            matrix, xs, ys = np.histogram2d(t, p, bins=[bin_edges(t, 48), bin_edges(p, 48)])
            chart["density"] = {
                "x": ((xs[:-1] + xs[1:]) / 2).tolist(),
                "y": ((ys[:-1] + ys[1:]) / 2).tolist(),
                "z": matrix.T.astype(int).tolist(),
            }
    if len(known):
        edges = bin_edges(numeric(known._truth), min(10, max(1, int(np.sqrt(len(known))))))
        counts = np.digitize(known._truth.to_numpy(), edges[1:-1])
        for i, part in known.groupby(counts):
            valid = part.loc[part._evaluated]
            j = int(i)
            chart["bins"].append(
                {
                    "lo": float(edges[j]),
                    "hi": float(edges[j + 1]),
                    "requested": len(part),
                    "evaluated": len(valid),
                    "metrics": scores(task, numeric(valid._truth), numeric(valid._pred)),
                }
            )
    lo_col, hi_col = method.get("lower_column"), method.get("upper_column")
    if lo_col and hi_col:
        lower, upper = numeric(scored[lo_col]), numeric(scored[hi_col])
        ok = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
        if ok.any():
            chart["interval"] = {
                "count": int(ok.sum()),
                "covered": int(((lower <= t) & (t <= upper) & ok).sum()),
                "mean_width": float((upper - lower)[ok].mean()),
                "nominal": method.get("nominal_coverage"),
                "width": histogram((upper - lower)[ok]),
            }
    chart["posterior"] = posterior_diagnostics(scored, method)
    chart["posterior_examples"] = native_posteriors(frame, method, has_id)
    return chart


def data_quality(frame: pd.DataFrame, method: Record) -> Record:
    modality = {
        name: boolean(frame[col], col) for name, col in method.get("modality_columns", {}).items()
    }
    counts: dict[str, int] = {}
    if modality:
        for flags, count in pd.DataFrame(modality).value_counts().items():
            flags = flags if isinstance(flags, tuple) else (flags,)
            label = " + ".join(n for n, enabled in zip(modality, flags, strict=True) if enabled)
            counts[label or "none"] = int(count)
    return {
        "modalities": counts,
        "missing_features": {
            col: int(frame[col].isna().sum()) for col in method.get("feature_columns", [])
        },
        "missing_slices": [
            col
            for col in [*method.get("slices", []), *method.get("slice_bins", {})]
            if col not in frame
        ],
    }


def load_history(method: Record) -> Any:
    history = method.get("history")
    if isinstance(history, (str, Path)):
        path = Path(history)
        if path.suffix == ".csv":
            history = pd.read_csv(path).to_dict("records")
        elif path.suffix == ".jsonl":
            history = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            history = json.loads(path.read_text())
    if isinstance(history, dict) and "history" in history:
        history = history["history"]
    if isinstance(history, list) and history:
        return (
            pd.json_normalize(history)
            .astype("object")
            .where(lambda f: f.notna(), None)
            .to_dict("records")
        )
    return history


def native_row(method: Record, task: ReportTask) -> tuple[Record, dict[str, pd.DataFrame]]:
    row: Record = {key: method[key] for key in ("name", "split")}
    row.update(
        status=method.get("status", "completed"),
        role=method.get("role", "unspecified"),
        history=None,
        details=method.get("details", {}),
        predictions=(
            str(method["predictions"])
            if isinstance(method.get("predictions"), (str, Path))
            else None
        ),
        slices={},
        paired_id=False,
    )
    for key in (
        "reason",
        "seed",
        "ensemble",
        "checkpoint",
        "duration",
        "version",
        "reference_kind",
    ):
        if key in method:
            row["details"] = {**row["details"], key: method[key]}
    if str(row["status"]).lower() in {
        "failed",
        "error",
        "skipped",
        "pending",
        "not_run",
        "planned",
    }:
        row.update(metrics=scores(task, np.array([]), np.array([])), coverage=None)
        return row, {}
    row["history"] = load_history(method)
    if method.get("predictions") is None:
        if row["status"].lower() in {"ok", "success", "complete", "completed"}:
            raise ValueError(f"Completed method {row['name']} has no predictions")
        row.update(metrics=scores(task, np.array([]), np.array([])), coverage=None)
        return row, {}
    frame, has_id = prepare_frame(method, task)
    if "image_split_overlap" in frame.attrs:
        row["details"] = {
            **row["details"], "image_split_overlap": frame.attrs["image_split_overlap"]
        }
    if has_id:
        frame = frame.sort_index()
    row["paired_id"] = has_id
    subsets = slice_frames(frame, method)
    for key, items in subsets.items():
        measured = items.loc[items._evaluated]
        row["slices"][key] = {
            "coverage": coverage(items),
            "metrics": scores(task, numeric(measured._truth), numeric(measured._pred)),
            "charts": diagnostics(items, task, method, has_id),
        }
    row["metrics"] = row["slices"]["All objects"]["metrics"]
    row["coverage"] = row["slices"]["All objects"]["coverage"]
    row["data"] = data_quality(frame, method)
    return row, subsets


def interval_for_delta(
    frame: pd.DataFrame,
    task: ReportTask,
    repeats: int,
) -> dict[str, list[float]] | None:
    if repeats < 2 or "_group" not in frame or frame._group.isna().any():
        return None
    groups = list(frame.groupby("_group", sort=True).indices.values())
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(0)
    truth, a, b = (numeric(frame[c]) for c in ("_truth", "_pred", "_other"))
    draws: dict[str, list[float]] = {key: [] for key in task.metrics}
    for _ in range(repeats):
        ix = np.concatenate([groups[int(i)] for i in rng.integers(len(groups), size=len(groups))])
        left, right = scores(task, truth[ix], a[ix]), scores(task, truth[ix], b[ix])
        for key in draws:
            x, y = left[key], right[key]
            if x is not None and y is not None:
                draws[key].append(x - y)
    return {k: np.percentile(values, [2.5, 97.5]).tolist() for k, values in draws.items() if values}


def examples(
    frame: pd.DataFrame, other: pd.DataFrame, task: ReportTask, method: Record, other_method: Record
) -> list[Record]:
    if frame.empty:
        return []
    truth, a, b = (numeric(frame[key]) for key in ("_truth", "_pred", "_other"))
    ra, rb = task.residual(truth, a), task.residual(truth, b)
    order = np.argsort(-np.abs(np.abs(ra) - np.abs(rb)), kind="stable")[:15]
    out = []
    for i in order:
        source = frame.iloc[i]
        optional = {col: source[col] for col in method.get("example_columns", []) if col in source}
        out.append(
            {
                "id": str(frame.index[i]),
                "truth": truth[i],
                "a": a[i],
                "b": b[i],
                "residual_a": ra[i],
                "residual_b": rb[i],
                "details": optional,
                "assets_a": example_assets(source, method),
                "assets_b": example_assets(other.loc[frame.index[i]], other_method),
            }
        )
    return out


def paired(
    a: pd.DataFrame,
    b: pd.DataFrame,
    task: ReportTask,
    method: Record,
    other_method: Record,
    repeats: int,
) -> Record:
    if not matching_id_types(a.index, b.index):
        raise ValueError("Cannot compare identities encoded as different types")
    common = a.index.intersection(b.index, sort=False)
    af, bf = a.loc[common], b.loc[common]
    compatible = np.isfinite(af._truth) & np.isfinite(bf._truth)
    if not np.allclose(af._truth[compatible], bf._truth[compatible], rtol=1e-6, atol=1e-6):
        raise ValueError("Paired IDs have inconsistent reference targets")
    if "_group" in af and "_group" in bf and not af._group.equals(bf._group):
        raise ValueError("Paired IDs have inconsistent group membership")
    good = af._evaluated & bf._evaluated
    frame = af.loc[good].assign(_other=bf.loc[good, "_pred"])
    t, x, y = (numeric(frame[key]) for key in ("_truth", "_pred", "_other"))
    left, right = scores(task, t, x), scores(task, t, y)
    delta = {}
    for key in task.metrics:
        x_score, y_score = left[key], right[key]
        delta[key] = x_score - y_score if x_score is not None and y_score is not None else None
    return {
        "shared_ids": len(common),
        "count": len(frame),
        "a": left,
        "b": right,
        "delta": delta,
        "ci95": interval_for_delta(frame, task, repeats),
        "examples": examples(frame, bf, task, method, other_method),
    }


def build_report_data(
    methods: list[Record],
    task: str | ReportTask,
    title: str,
    metadata: Record | None = None,
    *,
    strict: bool = False,
) -> Record:
    resolved = resolve_task(task)
    metadata = metadata or {}
    keys = [(m["split"], m["name"]) for m in methods]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Report requires unique method names within each split")
    evaluated = []
    for method in methods:
        try:
            evaluated.append(native_row(method, resolved))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            if strict:
                raise
            failed = dict(
                method, status="failed", reason=f"Evaluation: {type(exc).__name__}: {exc}"
            )
            evaluated.append(native_row(failed, resolved))
    rows, frames = zip(*evaluated, strict=True)
    pairs, overlaps = [], []
    for i, j in itertools.combinations(range(len(rows)), 2):
        a, b = rows[i], rows[j]
        have_ids = a["paired_id"] and b["paired_id"]
        if a["split"] != b["split"]:
            if a["name"] == b["name"] and have_ids:
                av, bv = frames[i]["All objects"], frames[j]["All objects"]
                groups = (
                    len(set(av._group.dropna()) & set(bv._group.dropna()))
                    if "_group" in av and "_group" in bv
                    else None
                )
                overlaps.append(
                    {
                        "name": a["name"],
                        "a": a["split"],
                        "b": b["split"],
                        "shared_ids": (
                            len(av.index.intersection(bv.index))
                            if matching_id_types(av.index, bv.index)
                            else None
                        ),
                        "shared_groups": groups,
                    }
                )
            continue
        result: Record = {"a": a["name"], "b": b["name"], "split": a["split"], "slices": {}}
        if (methods[i].get("reference_kind") or "truth") != (
            methods[j].get("reference_kind") or "truth"
        ):
            result["reason"] = "Роли эталона различаются: точность и согласие оцениваются отдельно."
        elif have_ids:
            for key in frames[i]:
                if key in frames[j]:
                    try:
                        result["slices"][key] = paired(
                            frames[i][key],
                            frames[j][key],
                            resolved,
                            methods[i],
                            methods[j],
                            int(metadata.get("bootstrap_replicates", 0)),
                        )
                    except ValueError as exc:
                        if strict:
                            raise
                        result["status"] = "failed"
                        result["reason"] = str(exc)
                        result["slices"] = {}
                        break
        else:
            result["reason"] = "Both methods need predictions with verified object IDs."
        pairs.append(result)
    seed_groups: dict[tuple[str, str], list[Record]] = {}
    for row in rows:
        details = row["details"]
        family = details.get("family") or details.get("recipe")
        if family and details.get("seed") is not None and row["coverage"] is not None:
            seed_groups.setdefault((row["split"], str(family)), []).append(row)
    seed_summaries = []
    for (split, family), members in seed_groups.items():
        if len(members) < 2:
            continue
        values = {}
        for metric in resolved.metrics:
            observed = [m["metrics"][metric] for m in members if m["metrics"][metric] is not None]
            if observed:
                values[metric] = {
                    "min": min(observed),
                    "max": max(observed),
                    "mean": float(np.mean(observed)),
                    "count": len(observed),
                }
        seed_summaries.append(
            {
                "split": split,
                "family": family,
                "methods": [m["name"] for m in members],
                "metrics": values,
            }
        )
    metric_defs = {
        k: {"label": m.label, "direction": m.direction, "unit": m.unit}
        for k, m in resolved.metrics.items()
    }
    return {
        "title": title,
        "task": resolved.name,
        "primary": resolved.primary,
        "metrics": metric_defs,
        "labels": {
            "truth": resolved.target_label,
            "pred": resolved.prediction_label,
            "residual": resolved.residual_label,
        },
        "rows": list(rows),
        "pairs": pairs,
        "overlaps": overlaps,
        "seed_summaries": seed_summaries,
        "metadata": metadata,
    }
