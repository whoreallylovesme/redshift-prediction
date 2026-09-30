#!/usr/bin/env python3
"""Оценка CapsNet photo-z на тест-тайлах Euclid EDF-N против PHZ-бейзлайна.

Загружает stage_b_best.pt, гонит инференс на штампах тест-тайлов, считает
NMAD/bias/долю выбросов против гризм-меток; те же метрики для PHZ_MEDIAN
на тех же объектах. Выход: preds parquet + metrics json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lib.dataloaders.euclid_capsnet import EuclidStampsDataset
from lib.models.capsnet import CapsNet, logistic_to_z, mdn_pz
from lib.utils.log import get_logger

logger = get_logger()


def predict(net: CapsNet, ds: EuclidStampsDataset, device: str,
            z_min: float, z_max: float, batch: int = 512,
            ) -> np.ndarray:
    """Точечный z: мода p(z) для MDN, логистика для точечной головы."""
    net.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            imgs = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))])
            z_out, _, _ = net(imgs.to(device))
            z_out = z_out.detach().cpu()
            if net.mdn_components > 0:
                grid = torch.linspace(z_min, z_max, 301)
                pz = mdn_pz(z_out, grid)
                out.append(grid[pz.argmax(dim=1)].numpy().astype(np.float64))
            else:
                z = logistic_to_z(z_out, z_min, z_max)
                out.append(np.asarray(z, dtype=np.float64))
    return np.concatenate(out)


def pz_metrics(net: CapsNet, ds: EuclidStampsDataset, z_true: np.ndarray, device: str,
               z_min: float, z_max: float) -> dict[str, float]:
    """CRPS и PIT-калиброванность (KS против равномерного)."""
    from scipy import stats

    grid = torch.linspace(z_min, z_max, 301)
    net.eval()
    crps_l: list[np.ndarray] = []
    pit_l: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            imgs = torch.stack([ds[j][0] for j in range(i, min(i + 512, len(ds)))])
            z_out, _, _ = net(imgs.to(device))
            pz = mdn_pz(z_out.detach().cpu(), grid).numpy()
            pdf = pz / pz.sum(axis=1, keepdims=True)
            cdf = np.cumsum(pdf, axis=1)
            zt = z_true[i : i + 512]
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
    dz = ((pd.Series(z_pred) - pd.Series(z_true)) / (1.0 + pd.Series(z_true))).to_numpy()
    return {
        "model": name,
        "n": int(len(z_true)),
        "nmad": float(1.4826 * np.median(np.abs(dz - np.median(dz)))),
        "bias": float(np.median(dz)),
        "outlier_frac": float(np.mean(np.abs(dz) > 0.15)),
    }


def main() -> None:
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
    args = ap.parse_args()

    labels = pd.read_parquet(args.labels_parquet)
    labels = labels[(labels["spe_gal_prob"] > 0.9) & (labels["spe_qso_prob"] < 0.1)]
    ds = EuclidStampsDataset(
        stamps_dir=Path(args.stamps_dir),
        tiles=list(args.test_tiles),
        labels=labels,
        norm_scales=np.ones(4, dtype=np.float32),
        z_min=args.z_min,
        z_max=args.z_max,
        augment=False,
    )
    scales_file = Path(args.checkpoint).parent / "norm_scales.npy"
    if scales_file.exists():
        ds.norm_scales = np.load(scales_file)
        logger.info(f"norm_scales из {scales_file}: {ds.norm_scales}")

    mdn_k = int(getattr(args, "mdn_components", 0))
    net = CapsNet(in_channels=4, input_size=64, num_class=2, dim_capsule=16, routings=3,
                  mdn_components=mdn_k)
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net = net.to(args.device)

    z_pred = predict(net, ds, args.device, args.z_min, args.z_max)
    z_true = ds.z

    rows = [metrics_row("capsnet_4ch", z_true, z_pred)]
    if net.mdn_components > 0:
        pz_row: dict[str, object] = {"model": "capsnet_4ch_pz"}
        pz_row.update(pz_metrics(net, ds, z_true, args.device, args.z_min, args.z_max))
        rows.append(pz_row)
    phz = labels.set_index("object_id").loc[ds.ids.tolist(), "phz_median"].to_numpy(float)
    mask = ~np.isnan(phz)
    if mask.any():
        rows.append(metrics_row("phz_median", z_true[mask], phz[mask]))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"object_id": ds.ids, "z_true": z_true, "z_pred": z_pred}).to_parquet(
        out / "preds_test.parquet", index=False,
    )
    (out / "metrics.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        logger.info(
            f"{r['model']}: NMAD={r['nmad']:.4f} bias={r['bias']:+.4f} "
            f"outliers={r['outlier_frac']:.3f} n={r['n']}",
        )


if __name__ == "__main__":
    main()
