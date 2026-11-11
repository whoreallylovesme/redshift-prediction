from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure
from numpy.typing import NDArray

from lib.metrics.bootstrap import photoz_stats

ML, PPUB, PFREE, POFF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TRAIN_C, TEST_C, HDX_C = "#6250d6", "#008300", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
NAMES = {"ml": "Наша модель", "ppub": "Phosphoros", "zpf": "Phosphoros без N", "poff": "PHZ Euclid"}
COL = {"ml": ML, "ppub": PPUB, "zpf": PFREE, "poff": POFF}
MRK = {"ml": "o", "ppub": "s", "zpf": "D", "poff": "^"}
LS = {"ml": "-", "ppub": "-", "zpf": "--", "poff": ":"}
CLASSIC = {
    "catboost": "CatBoost",
    "rf": "Random Forest",
    "annz": "ANNz",
    "llr": "LLR (k-NN)",
    "gam": "GAM",
}
STYLE: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}
MODE_EDGES = np.round(np.arange(0, 1.81, 0.15), 3)
BIN_Z = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
BIN_VIS = [17, 19, 20, 21, 22, 22.5, 23, 23.5, 24.5]
MIN_VIOLIN = 30
MIN_BIN = 50
MIN_MODE = 15

type Sample = tuple[NDArray[np.float64], NDArray[np.float64]]


def fmt(x: int) -> str:
    return f"{x:,}".replace(",", "\u202f")


def fig_dist(out: Path, labels: Mapping[str, Sample]) -> dict[str, dict[str, float]]:
    colors = {"F": TRAIN_C, "S": HDX_C, "N": TEST_C}
    with rc_context(STYLE):
        fig = Figure(figsize=(6.7, 2.4))
        ax = fig.subplots(1, 2)
        bz, bv = np.linspace(0, 3, 61), np.linspace(16, 26, 51)
        for k in ("F", "S", "N"):
            z, v = labels[k]
            z = z[np.isfinite(z) & (z >= 0)]
            v = v[np.isfinite(v)]
            ax[0].hist(
                z,
                bins=bz,
                density=True,
                histtype="step",
                color=colors[k],
                lw=1.5,
                label=f"{k}, n={fmt(len(z))}",
            )
            ax[1].hist(v, bins=bv, density=True, histtype="step", color=colors[k], lw=1.5)
        ax[0].set_xlabel("z спектральное")
        ax[0].set_ylabel("плотность")
        ax[0].set_xlim(0, 3)
        ax[0].legend(loc="upper right")
        ax[1].set_xlabel("VIS, звёздная величина (AB)")
        ax[1].set_xlim(16, 26)
        for a, t in zip(ax, ("a", "b"), strict=True):
            a.set_title(f"({t})", loc="left", color=INK2)
        fig.savefig(out / "fig_dist.pdf")
    stats: dict[str, dict[str, float]] = {}
    for k, (z, _) in labels.items():
        fin = np.isfinite(z)
        stats[k] = {"n_z": int(fin.sum()), "frac_z_gt1": float(np.mean(z[fin] > 1))}
    return stats


def fig_scatter(out: Path, d: pd.DataFrame, name: str, fname: str, zmax: float) -> None:
    with rc_context(STYLE):
        fig = Figure(figsize=(6.7, 3.05), constrained_layout=True)
        ax = fig.subplots(1, 2, sharey=True)
        norm = LogNorm(1, None)
        hexes = []
        for a, key in zip(ax, ("ml", "ppub"), strict=True):
            hexes.append(
                a.hexbin(
                    d.z,
                    d[key],
                    gridsize=70,
                    extent=(0, zmax, 0, zmax),
                    cmap=BLUES,
                    norm=norm,
                    mincnt=1,
                    linewidths=0,
                )
            )
            x = np.linspace(0, zmax, 50)
            a.plot(x, x, color=INK, lw=0.8)
            for s in (1, -1):
                a.plot(x, x + s * 0.15 * (1 + x), color=INK2, lw=0.7, ls="--")
            n, o, b = photoz_stats(d[key].values, d.z.values)
            a.text(
                0.03,
                0.97,
                f"{NAMES[key]}\nNMAD {n:.4f}\nвыбросы {100 * o:.1f}%\nbias {b:+.4f}",
                transform=a.transAxes,
                va="top",
                ha="left",
                fontsize=7.5,
                color=INK,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": GRID, "lw": 0.6},
            )
            a.set_xlim(0, zmax)
            a.set_ylim(0, zmax)
            a.set_xlabel("z спектральное")
            a.set_aspect("equal")
            a.grid(False)
        ax[0].set_ylabel("z фотометрическое")
        cb = fig.colorbar(hexes[-1], ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("объектов в ячейке", color=INK2)
        outline: Any = cb.outline
        outline.set_linewidth(0.4)
        fig.suptitle(f"{name}: n = {fmt(len(d))}", fontsize=9, color=INK)
        fig.savefig(out / fname)


def mode_of(v: NDArray[np.float64], lo: float, hi: float) -> float:
    if len(v) < MIN_MODE:
        return np.nan
    h, e = np.histogram(v, bins=np.arange(lo, hi + 1e-9, 0.01))
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2)
    hs = np.convolve(h, k / k.sum(), mode="same")
    i = int(np.argmax(hs))
    return float(0.5 * (e[i] + e[i + 1]))


def fig_modes(out: Path, d: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    cen = 0.5 * (MODE_EDGES[1:] + MODE_EDGES[:-1])
    rec: dict[str, dict[str, list[float]]] = {}
    with rc_context(STYLE):
        fig = Figure(figsize=(6.7, 3.4))
        ax = fig.subplots()
        w = 0.055
        for key, off in (("ml", -0.032), ("ppub", 0.032)):
            modes, meds = [], []
            for lo, hi, c in zip(MODE_EDGES[:-1], MODE_EDGES[1:], cen, strict=True):
                v = d[key].values[(d.z.values >= lo) & (d.z.values < hi)]
                v = v[(v >= 0) & (v <= 3)]
                if len(v) >= MIN_VIOLIN:
                    vp = ax.violinplot(
                        [v], positions=[c + off], widths=w, showextrema=False, points=200
                    )
                    for body in cast(list[PolyCollection], vp["bodies"]):
                        body.set_facecolor(COL[key])
                        body.set_edgecolor("none")
                        body.set_alpha(0.28)
                modes.append(mode_of(v, 0, 3))
                meds.append(float(np.median(v)) if len(v) else np.nan)
            ax.plot(
                cen + off,
                np.array(modes),
                color=COL[key],
                marker=MRK[key],
                lw=1.5,
                ms=4.5,
                label=f"{NAMES[key]}: мода по бину",
            )
            rec[key] = {"bin_centres": cen.tolist(), "mode": modes, "median": meds}
        ax.plot([0, 1.8], [0, 1.8], color=INK, lw=0.8, ls="--", label="z_phot = z_spec")
        ax.set_xlim(0, 1.8)
        ax.set_ylim(0, 2.2)
        ax.set_xlabel("z спектральное (центр бина, шаг 0.15)")
        ax.set_ylabel("z фотометрическое")
        ax.legend(loc="upper left")
        fig.savefig(out / "fig_modes.pdf")
    return rec


def fig_bins(out: Path, d: pd.DataFrame) -> dict[str, dict[str, object]]:
    rec: dict[str, dict[str, object]] = {}
    with rc_context(STYLE):
        fig = Figure(figsize=(6.7, 4.4))
        ax = fig.subplots(2, 2, sharex="col")
        for j, (col, edges, xl) in enumerate(
            (("z", BIN_Z, "z спектральное"), ("vis", BIN_VIS, "VIS (AB)"))
        ):
            c = 0.5 * (np.array(edges[1:]) + np.array(edges[:-1]))
            rec[col] = {"edges": edges}
            for key in ("ml", "ppub", "poff"):
                nm: list[float] = []
                ou: list[float] = []
                for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                    m = (d[col] >= lo) & (d[col] < hi)
                    if m.sum() < MIN_BIN:
                        nm.append(np.nan)
                        ou.append(np.nan)
                        continue
                    n, o, _ = photoz_stats(d[key].values[m], d.z.values[m])
                    nm.append(n)
                    ou.append(100 * o)
                ax[0, j].plot(c, nm, color=COL[key], marker=MRK[key], ls=LS[key], label=NAMES[key])
                ax[1, j].plot(c, ou, color=COL[key], marker=MRK[key], ls=LS[key])
                rec[col][key] = {"nmad": nm, "outliers_pct": ou}
            ax[1, j].set_xlabel(xl)
        ax[0, 0].set_ylabel("NMAD")
        ax[1, 0].set_ylabel("выбросы, %")
        for a, t in zip(ax.flat, "abcd", strict=True):
            a.set_title(f"({t})", loc="left", color=INK2)
            a.set_ylim(bottom=0)
        ax[0, 0].legend(loc="upper left", ncol=1)
        fig.savefig(out / "fig_bins.pdf")
    return rec


def fig_baselines(out: Path, ns: Mapping[str, Any]) -> None:
    keys: list[str] = list(ns["keys"])
    label = NAMES | {k: CLASSIC[k] for k in keys}
    color = COL | {k: INK2 for k in keys}
    r = ns["ALL"]
    ks = sorted(["ml", "ppub", "poff", *keys], key=lambda k: r[k]["nmad"])
    with rc_context(STYLE):
        fig = Figure(figsize=(6.7, 3.0))
        ax = fig.subplots()
        y = np.arange(len(ks))[::-1]
        for yy, k in zip(y, ks, strict=True):
            v, (lo, hi) = r[k]["nmad"], r[k]["nmad_ci"]
            ax.errorbar(
                v,
                yy,
                xerr=[[v - lo], [hi - v]],
                fmt="o" if k in COL else "s",
                color=color[k],
                ms=4,
                lw=1.2,
                capsize=0,
            )
        ax.set_yticks(y)
        ax.set_yticklabels([label[k] for k in ks], fontsize=7.5)
        ax.set_title(f"N, все источники (n={fmt(r['n'])})", fontsize=8.5, color=INK)
        ax.set_xlabel("NMAD, 95% ДИ")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(out / "fig_baselines.pdf")


def fig_train(
    out: Path,
    clip: Mapping[str, Sequence[Mapping[str, Any]]],
    fine: Mapping[str, Sequence[Mapping[str, Any]]],
    val_ppub: float,
    val_poff: float,
) -> dict[str, dict[str, list[float]]]:
    shades = ["#86b6ef", "#2a78d6", "#104281"]
    rec: dict[str, dict[str, list[float]]] = {}
    with rc_context(STYLE):
        fig = Figure(figsize=(6.7, 2.4))
        ax = fig.subplots(1, 3)
        for i, (seed, history) in enumerate(clip.items()):
            ep, loss = [x["epoch"] for x in history], [x["loss"] for x in history]
            ax[0].plot(
                ep, loss, color=shades[2 * i % len(shades)], marker="o", ms=3, label=f"сид {seed}"
            )
            rec[f"clip_{seed}"] = {"epoch": ep, "loss": loss}
        ax[0].set_xlabel("эпоха")
        ax[0].set_xticks(range(2, 13, 2))
        ax[0].set_ylabel("InfoNCE")
        ax[0].set_title("(a) предобучение CLIP", loc="left", color=INK2)
        ax[0].legend(loc="upper right")
        last = 0
        for j, (name, history) in enumerate(fine.items()):
            ep = [x["epoch"] for x in history]
            nll = [x["train_nll"] for x in history]
            nmad = [x["valid"]["nmad"] for x in history]
            c = shades[j % len(shades)]
            ax[1].plot(ep, nll, color=c, lw=1.0, marker="o", ms=2.5)
            ax[2].plot(ep, nmad, color=c, lw=1.0, marker="o", ms=2.5)
            rec[name] = {"epoch": ep, "train_ce": nll, "fvalid_nmad": nmad}
            last = max(last, *ep)
        ax[1].set_xlabel("эпоха")
        ax[1].set_ylabel("кросс-энтропия (трейн)")
        ax[1].set_title(f"(b) дообучение, {len(fine)} сидов", loc="left", color=INK2)
        vo, vp = val_poff, val_ppub
        ax[2].axhline(vo, color=POFF, ls=":", lw=1.2)
        hi_, lo_ = max(vo, vp), min(vo, vp)
        ax[2].axhline(vp, color=PPUB, ls="-", lw=1.0)
        upper = f"PHZ Euclid {vo:.4f}" if vo >= vp else f"Phosphoros {vp:.4f}"
        lower = f"Phosphoros {vp:.4f}" if vo >= vp else f"PHZ Euclid {vo:.4f}"
        ax[2].text(last, hi_ + 0.0008, upper, color=INK2, fontsize=7, ha="right", va="bottom")
        ax[2].text(last, lo_ - 0.0008, lower, color=INK2, fontsize=7, ha="right", va="top")
        ax[2].set_xlabel("эпоха")
        ax[2].set_ylabel("NMAD (валидация)")
        ax[2].set_title(f"(c) валидация F+S, {len(fine)} сидов", loc="left", color=INK2)
        ax[2].set_ylim(0, max(vo, vp) * 1.25)
        fig.tight_layout()
        fig.savefig(out / "fig_train.pdf")
    return rec
