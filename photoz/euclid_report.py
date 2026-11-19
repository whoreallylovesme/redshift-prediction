from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.report import render_report
from lib.report.render import json_value
from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as
from photoz.artifacts import HISTORY_NAME
from photoz.bootstrap import Record, stats_record
from photoz.data.euclid_report import (
    HETDEX_SET,
    HETDEX_SOURCE,
    NORTH_SETS,
    NORTH_SOURCE,
    VALID_SET,
    ZVAL_SET,
    ReportConfig,
    SeedStack,
    classical_frames,
    classical_series,
    eval_blocks,
    field_labels,
    hetdex_frame,
    hetdex_inputs,
    label_facts,
    load_metas,
    load_predictions,
    north_frame,
    north_table,
    phosphoros_points,
    phosphoros_series,
    run_names,
    sources_frame,
    zval_inputs,
)
from photoz.euclid_eval import (
    COMPARATORS,
    classical_metrics,
    comparison_results,
    hetdex_results,
    source_metrics,
    zval_metrics,
)
from photoz.figures import (
    CLASSIC,
    fig_baselines,
    fig_bins,
    fig_dist,
    fig_modes,
    fig_scatter,
    fig_train,
)
from photoz.report import PHOTOZ

logger = get_logger()

SOURCES = (NORTH_SOURCE, "NEP-Wide", HETDEX_SOURCE)
TABLE_METHODS = ("ml", "ppub", "poff")
TABLE_ROWS = (
    ("\\textbf{N, все источники}", "ALL"),
    ("\\quad то же, $z\\ge1$", "ALL_z1"),
    ("N: DESI", "DESI"),
    ("N: NEP-Wide", "NEP-Wide"),
    ("N: HETDEX [OII]", "HETDEX"),
)
CLASSICAL_COLUMNS = ("ALL", "DESI", "NEP-Wide", "HETDEX")
BASE_LABELS = (
    ("ml", "\\textcolor{ml}{Наша модель}"),
    ("ppub", "\\textcolor{ppub}{Phosphoros}"),
    ("poff", "PHZ Euclid"),
)
VALID_LABEL = "F+S: валидация"
SCATTER_ZMAX = 2.0


def cell(nmad: float, outliers: float) -> str:
    if not np.isfinite([nmad, outliers]).all():
        return "—"
    return f"{nmad:.4f} / {100 * outliers:.1f}\\%"


def count(x: int) -> str:
    return f"{x:,}".replace(",", "\\,")


def delta_cell(v: Mapping[str, Any]) -> str:
    lo, hi = v["ci"]
    if not np.isfinite([v["d_nmad"], lo, hi]).all():
        return "—"
    return f"${v['d_nmad']:+.4f}$ [${lo:+.4f}$; ${hi:+.4f}$]"


def metrics_tables(val: Mapping[str, Any], ns: Mapping[str, Any]) -> tuple[str, str]:
    cells = [cell(val[m]["nmad"], val[m]["outliers"]) for m in TABLE_METHODS]
    rows = [" & ".join([VALID_LABEL, count(val["n"]), *cells]) + " \\\\", "\\midrule"]
    deltas = []
    for label, key in TABLE_ROWS:
        r = ns[key]
        cells = [cell(r[m]["nmad"], r[m]["outliers"]) for m in TABLE_METHODS]
        rows.append(" & ".join([label, count(r["n"]), *cells]) + " \\\\")
        d = [delta_cell(r[f"d_vs_{m}"]) for m in TABLE_METHODS[1:]]
        deltas.append(" & ".join([label, *d]) + " \\\\")
    return "\n".join(rows) + "\n", "\n".join(deltas) + "\n"


def classical_table(ns: Mapping[str, Any]) -> str:
    rows = []
    for key, label in [*BASE_LABELS, *((k, CLASSIC[k]) for k in ns["keys"])]:
        cells = [cell(ns[s][key]["nmad"], ns[s][key]["outliers"]) for s in CLASSICAL_COLUMNS]
        d = "—" if key == "ml" else delta_cell(ns["ALL"][f"d_vs_{key}"])
        rows.append(" & ".join([label, *cells, d]) + " \\\\")
        if key == "poff":
            rows.append("\\midrule")
    return "\n".join(rows) + "\n"


def write_json(path: Path, rec: Mapping[str, Any], as_float: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_value(dict(rec)), indent=1, allow_nan=False,
                      default=float if as_float else None)
    path.write_text(text + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def interactive_methods(
    cfg: ReportConfig,
    eligible: pd.DataFrame,
    preds: Mapping[str, SeedStack],
    histories: Mapping[str, Any],
    zval: np.ndarray,
    val_methods: Mapping[str, np.ndarray],
    field_validation: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    names = run_names(cfg.tag, cfg.seeds)
    labels = {
        "ml": f"{cfg.tag} · ensemble",
        "ppub": "phosphoros_public",
        "zpf": "phosphoros_zpf",
        "poff": "euclid_phz",
        **{k: CLASSIC[k] for k in cfg.learners},
    }
    groups = [(str(k), f.copy()) for k, f in eligible.groupby("_set", sort=False)]
    if field_validation is not None:
        groups.append((VALID_SET, field_validation.copy()))
    groups.append(
        (ZVAL_SET, pd.DataFrame({"object_id": preds[ZVAL_SET].ids, "z": zval, **val_methods}))
    )
    methods = []
    for group, frame in groups:
        development = group in {VALID_SET, ZVAL_SET}
        field = "F" if group == VALID_SET else "F+S" if group == ZVAL_SET else "N"
        stack_keys = NORTH_SETS if group == "north + nb" else (group,)
        for i, name in enumerate(names):
            series = pd.concat([pd.Series(preds[k].z[i], index=preds[k].ids) for k in stack_keys])
            frame[name] = series.reindex(frame.object_id).to_numpy()
        if not development:
            frame["z"] = frame.z.where(frame.z > 0)
        for column, label in {**labels, **{n: n for n in names}}.items():
            if column not in frame:
                continue
            details: dict[str, Any] = {"recipe": cfg.tag}
            if column == "ml":
                details["ensemble"] = names
            if column in names:
                details["seed"] = sorted(cfg.seeds)[names.index(column)]
            if column in ("ppub", "zpf"):
                resource = cfg.phosphoros if column == "ppub" else cfg.phosphoros_zpf
                details["resources"] = resource.model_dump(mode="json")
            methods.append(
                {
                    "name": label,
                    "split": f"{field} · {group}",
                    "role": "development" if development else "held-out",
                    "predictions": frame,
                    "id_column": "object_id",
                    "truth_column": "z",
                    "prediction_column": column,
                    "history": histories.get(column),
                    "details": details,
                    "group_column": "tile" if "tile" in frame else None,
                    "slices": ["src"] if "src" in frame else [],
                    "slice_bins": {"vis": [0, 21, 22.5, 30]}
                    if "vis" in frame and frame.vis.notna().any()
                    else {},
                }
            )
    return methods


def build_report(config_path: Path) -> None:
    cfg = load_yaml_as(str(config_path), ReportConfig)
    check(set(cfg.learners) <= set(CLASSIC), f"classical learners within {sorted(CLASSIC)}")
    figures = cfg.out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    preds = load_predictions(cfg.pred_dir, cfg.tag, cfg.seeds)
    metas = load_metas(cfg.meta_dir, cfg.tag, cfg.seeds)
    nt = north_table(cfg.north_labels)
    pub = phosphoros_series(cfg.phosphoros.files)
    zpf = phosphoros_series(cfg.phosphoros_zpf.files)
    north = north_frame(preds, nt, pub, zpf, include_incomplete=True)
    hetdex = hetdex_frame(cfg, preds[HETDEX_SET], include_incomplete=True)
    n = north.loc[np.isfinite(north[["z", *COMPARATORS]]).all(axis=1)]
    hx = hetdex.loc[np.isfinite(hetdex[["z", "ml", "ppub", "poff"]]).all(axis=1) & (hetdex.z > 0)]
    logger.info(json.dumps({"dist": fig_dist(figures, field_labels(cfg, hx))}))
    fig_scatter(figures, n, "Поле N, надёжные метки DESI", "fig_scatter_N.pdf", SCATTER_ZMAX)
    classical = {
        k: classical_series(cfg.classical_dir / k / "predictions.npz") for k in cfg.learners
    }
    cl_frame, cl_sets = classical_frames(cfg, n, hx, preds, nt, classical)
    cl = classical_metrics(cl_frame, cfg.learners, cl_sets)
    eligible = sources_frame(
        cfg, north, hetdex, preds, nt, pub, zpf, classical, include_incomplete=True
    )
    columns = ["z", *COMPARATORS, *cfg.learners]
    dall = eligible.loc[np.isfinite(eligible[columns]).all(axis=1) & (eligible.z > 0)]
    ns = source_metrics(dall, [*COMPARATORS, *cfg.learners], SOURCES)
    ns["keys"] = list(cfg.learners)
    fig_scatter(figures, dall, "Поле N, все метки", "fig_scatter_Nall.pdf", SCATTER_ZMAX)
    fig_baselines(figures, ns)
    first = metas[next(iter(metas))]
    rec: Record = {
        "modes": fig_modes(figures, n),
        "bins": fig_bins(figures, n),
        "facts": label_facts(cfg, first, int(len(n)), int(len(hx))),
        "classical": cl,
        "N_sources": ns,
    }
    zval, val_methods = zval_inputs(cfg, metas, preds[ZVAL_SET])
    val = zval_metrics(zval, val_methods)
    rec["VAL"] = val
    clip = {
        str(seed): json.loads((cfg.clip_dir / f"clip_1m_{seed}" / HISTORY_NAME).read_text())
        for seed in cfg.clip_seeds
    }
    fine = {name: meta["history"] for name, meta in metas.items()}
    rec["training"] = fig_train(figures, clip, fine, val["ppub"]["nmad"], val["poff"]["nmad"])
    rec["N_common"] = {k: stats_record(n[k].values, n.z.values) for k in COMPARATORS}
    rec["HETDEX_primary"] = {k: stats_record(hx[k].values, hx.z.values) for k in ("ml", "ppub")}
    write_json(cfg.out_dir / "facts.json", rec, as_float=True)
    metrics, deltas = metrics_tables(val, ns)
    write_text(cfg.out_dir / "tables" / "metrics.tex", metrics)
    write_text(cfg.out_dir / "tables" / "metrics_delta.tex", deltas)
    write_text(cfg.out_dir / "tables" / "classical.tex", classical_table(ns))
    blocks = eval_blocks(cfg, preds, nt)
    for name, source in (("pub", cfg.phosphoros), ("zpf", cfg.phosphoros_zpf)):
        points, widths = phosphoros_points(source.files)
        results = comparison_results(cfg.tag, preds, blocks, points, widths)
        out = cfg.out_dir / "metrics" / f"results_{cfg.run_name}_vs_{name}.json"
        write_json(out, results, as_float=False)
    h6 = hetdex_results(cfg.tag, preds[HETDEX_SET], *hetdex_inputs(cfg, preds[HETDEX_SET]))
    write_json(cfg.out_dir / "metrics" / f"results_h6_{cfg.tag}.json", h6, as_float=True)
    _, valid_methods, valid_truth, valid_tiles = next(s for s in cl_sets if s[0] == "F_VALID")
    valid_block = next(b for b in blocks if b.name == "F_VALID")
    ids = preds[VALID_SET].ids
    valid = pd.DataFrame({
        "object_id": ids, "z": valid_truth, "tile": valid_tiles, **valid_methods,
        "ppub": pub.reindex(ids).to_numpy(), "zpf": zpf.reindex(ids).to_numpy(),
        "poff": valid_block.phz,
    })
    methods = interactive_methods(cfg, eligible, preds, fine, zval, val_methods, valid)
    interactive = render_report(
        cfg.out_dir,
        methods,
        PHOTOZ,
        f"Euclid Q1 · {cfg.tag}",
        {
            "config": cfg.model_dump(mode="json"),
            "config_file": str(config_path),
            "pretraining_histories": clip,
            "legacy_common_sample_comparisons": ns,
            "scientific_facts_link": "facts.json",
            "metrics_table_link": "tables/metrics.tex",
        },
    )
    failures = [r["name"] for r in interactive["rows"] if r["status"] == "failed"]
    failures.extend(
        f"{p['a']} / {p['b']}" for p in interactive["pairs"] if p.get("status") == "failed"
    )
    if failures:
        raise ValueError(f"Report saved with evaluation failures: {', '.join(failures)}")
    logger.info(
        json.dumps(
            {"N": int(len(n)), "HETDEX": int(len(hx)), "N_common": rec["N_common"]}, default=float
        )
    )
