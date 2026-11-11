from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lib.dataloaders.euclid_report import (
    HETDEX_SET,
    HETDEX_SOURCE,
    NORTH_SOURCE,
    ZVAL_SET,
    ReportConfig,
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
    sources_frame,
    zval_inputs,
)
from lib.metrics.bootstrap import Record, stats_record
from lib.metrics.euclid_eval import (
    COMPARATORS,
    classical_metrics,
    comparison_results,
    hetdex_results,
    source_metrics,
    zval_metrics,
)
from lib.train.euclid_clip_pretrain import HISTORY_NAME
from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as
from lib.visual.euclid_report import (
    CLASSIC,
    fig_baselines,
    fig_bins,
    fig_dist,
    fig_modes,
    fig_scatter,
    fig_train,
)

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
    return f"{nmad:.4f} / {100 * outliers:.1f}\\%"


def count(x: int) -> str:
    return f"{x:,}".replace(",", "\\,")


def delta_cell(v: Mapping[str, Any]) -> str:
    lo, hi = v["ci"]
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
    text = json.dumps(rec, indent=1, default=float) if as_float else json.dumps(rec, indent=1)
    path.write_text(text + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    n = north_frame(preds, nt, pub, zpf)
    hx = hetdex_frame(cfg, preds[HETDEX_SET])
    logger.info(json.dumps({"dist": fig_dist(figures, field_labels(cfg, hx))}))
    fig_scatter(
        figures, n, "Поле N, надёжные метки DESI", "fig_scatter_N.pdf", SCATTER_ZMAX
    )
    classical = {
        k: classical_series(cfg.classical_dir / k / "predictions.npz") for k in cfg.learners
    }
    cl_frame, cl_sets = classical_frames(cfg, n, hx, preds, nt, classical)
    cl = classical_metrics(cl_frame, cfg.learners, cl_sets)
    dall = sources_frame(cfg, n, hx, preds, nt, pub, zpf, classical)
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
    val = zval_metrics(*zval_inputs(cfg, metas, preds[ZVAL_SET]))
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
    logger.info(
        json.dumps(
            {"N": int(len(n)), "HETDEX": int(len(hx)), "N_common": rec["N_common"]}, default=float
        )
    )
