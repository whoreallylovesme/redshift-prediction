#!/usr/bin/env python3
"""Оценка CapsNet photo-z на тест-тайлах Euclid EDF-N против PHZ-бейзлайна.

Загружает stage_b_best.pt, гонит инференс на штампах тест-тайлов, считает
NMAD/bias/долю выбросов против гризм-меток; те же метрики для PHZ_MEDIAN
на тех же объектах. Выход: preds parquet + metrics json.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lib.dataloaders.euclid_capsnet import EuclidStampsDataset
from lib.metrics import bias, delta, nmad
from lib.models.capsnet import CapsNet, logistic_to_z, mdn_pz
from lib.utils.log import get_logger

logger = get_logger()

# Общая сетка p(z) в логит-пространстве для predict и pz_metrics.
_PZ_GRID_LOGIT = torch.linspace(0.01, 0.99, 301)


def _iter_z_out(net: CapsNet, ds: EuclidStampsDataset, device: str,
                batch: int = 512) -> Iterator[tuple[torch.Tensor, int, int]]:
    """Батчевый инференс: z_out на CPU плюс [start, stop) индексов объектов."""
    net.eval()
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            stop = min(i + batch, len(ds))
            imgs = torch.stack([ds[j][0] for j in range(i, stop)])
            z_out, _, _ = net(imgs.to(device))
            yield z_out.detach().cpu(), i, stop


def predict(net: CapsNet, ds: EuclidStampsDataset, device: str,
            z_min: float, z_max: float, batch: int = 512,
            ) -> np.ndarray:
    """Точечный z: мода p(z) для MDN, логистика для точечной головы."""
    z_vals = logistic_to_z(_PZ_GRID_LOGIT, z_min, z_max)
    out: list[np.ndarray] = []
    for z_out, _start, _stop in _iter_z_out(net, ds, device, batch):
        if net.mdn_components > 0:
            pz = mdn_pz(z_out, _PZ_GRID_LOGIT)
            out.append(z_vals[pz.argmax(dim=1)].numpy().astype(np.float64))
        else:
            z = logistic_to_z(z_out, z_min, z_max)
            out.append(np.asarray(z, dtype=np.float64))
    return np.concatenate(out)


def pz_metrics(net: CapsNet, ds: EuclidStampsDataset, z_true: np.ndarray, device: str,
               z_min: float, z_max: float) -> dict[str, float]:
    """CRPS и PIT-калиброванность (KS против равномерного)."""
    from scipy import stats


    grid = logistic_to_z(_PZ_GRID_LOGIT, z_min, z_max)
    crps_l: list[np.ndarray] = []
    pit_l: list[np.ndarray] = []
    for z_out, start, stop in _iter_z_out(net, ds, device):
        pz = mdn_pz(z_out, grid).numpy()
        pdf = pz / pz.sum(axis=1, keepdims=True)
        cdf = np.cumsum(pdf, axis=1)
        zt = z_true[start:stop]
        g = grid.numpy()
        cdf_at = np.array([np.interp(t, g, c) for c, t in zip(cdf, zt, strict=False)])
        step = (g[None, :] <= zt[:, None]).astype(float)
        crps_l.append(np.trapezoid((cdf - step) ** 2, g, axis=1))
        pit_l.append(cdf_at)
    crps_arr = np.concatenate(crps_l)
    pit_arr = np.concatenate(pit_l)
    ks = float(stats.kstest(pit_arr, "uniform").statistic)
    return {"crps_mean": float(crps_arr.mean()), "pit_ks_uniform": ks}


def metrics_row(name: str, z_true: np.ndarray, z_pred: np.ndarray) -> dict[str, object]:
    y_true = pd.Series(z_true)
    return {
        "model": name,
        "n": int(len(z_true)),
        "nmad": nmad(y_true, z_pred),
        "bias": bias(y_true, z_pred),
        # Строгое '>', в отличие от lib.outlier_fraction ('>='): делегирование
        # меняло бы граничное сравнение, поэтому считаем локально от lib.delta.
        "outlier_frac": float(np.mean(np.abs(delta(y_true, z_pred)) > 0.15)),
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stamps-dir", required=True)
    ap.add_argument("--labels-parquet", required=True)
    ap.add_argument("--test-tiles", type=int, nargs="+", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--z-min", type=float, default=0.0)
    ap.add_argument("--z-max", type=float, default=3.0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--mdn-components", type=int, default=0)
    # Архитектура CapsNet (должна совпадать с чекпоинтом).
    ap.add_argument("--in-channels", type=int, default=4)
    ap.add_argument("--input-size", type=int, default=64)
    ap.add_argument("--num-class", type=int, default=2)
    ap.add_argument("--dim-capsule", type=int, default=16)
    ap.add_argument("--routings", type=int, default=3)
    return ap.parse_args()


def _load_test_labels(labels_parquet: str) -> pd.DataFrame:
    labels = pd.read_parquet(labels_parquet)
    return labels[(labels["spe_gal_prob"] > 0.9) & (labels["spe_qso_prob"] < 0.1)]


def _build_dataset(args: argparse.Namespace, labels: pd.DataFrame) -> EuclidStampsDataset:
    ds = EuclidStampsDataset(
        stamps_dir=Path(args.stamps_dir),
        tiles=list(args.test_tiles),
        labels=labels,
        norm_scales=np.ones(args.in_channels, dtype=np.float32),
        z_min=args.z_min,
        z_max=args.z_max,
        augment=False,
    )
    scales_file = Path(args.checkpoint).parent / "norm_scales.npy"
    if scales_file.exists():
        ds.norm_scales = np.load(scales_file)
        logger.info(f"norm_scales из {scales_file}: {ds.norm_scales}")
    return ds


def _load_net(args: argparse.Namespace) -> CapsNet:
    net = CapsNet(
        in_channels=args.in_channels,
        input_size=args.input_size,
        num_class=args.num_class,
        dim_capsule=args.dim_capsule,
        routings=args.routings,
        mdn_components=args.mdn_components,
    )
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net = net.to(args.device)
    return net


def _collect_rows(net: CapsNet, ds: EuclidStampsDataset, labels: pd.DataFrame,
                  z_true: np.ndarray, z_pred: np.ndarray, device: str,
                  z_min: float, z_max: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [metrics_row("capsnet_4ch", z_true, z_pred)]
    if net.mdn_components > 0:
        pz_row: dict[str, object] = {"model": "capsnet_4ch_pz"}
        pz_row.update(pz_metrics(net, ds, z_true, device, z_min, z_max))
        rows.append(pz_row)
    phz = labels.set_index("object_id").loc[ds.ids.tolist(), "phz_median"].to_numpy(float)
    mask = ~np.isnan(phz)
    if mask.any():
        rows.append(metrics_row("phz_median", z_true[mask], phz[mask]))
    return rows


def _write_outputs(out_dir: Path, ids: np.ndarray, z_true: np.ndarray,
                   z_pred: np.ndarray, rows: list[dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"object_id": ids, "z_true": z_true, "z_pred": z_pred}).to_parquet(
        out_dir / "preds_test.parquet", index=False,
    )
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=1))


def _log_rows(rows: list[dict[str, object]]) -> None:
    for r in rows:
        logger.info(
            f"{r['model']}: NMAD={r['nmad']:.4f} bias={r['bias']:+.4f} "
            f"outliers={r['outlier_frac']:.3f} n={r['n']}",
        )


def main() -> None:
    args = _parse_args()
    labels = _load_test_labels(args.labels_parquet)
    ds = _build_dataset(args, labels)
    net = _load_net(args)

    z_pred = predict(net, ds, args.device, args.z_min, args.z_max)
    z_true = ds.z

    rows = _collect_rows(net, ds, labels, z_true, z_pred, args.device, args.z_min, args.z_max)
    _write_outputs(Path(args.out_dir), ds.ids, z_true, z_pred, rows)
    _log_rows(rows)


if __name__ == "__main__":
    main()
