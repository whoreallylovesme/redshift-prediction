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
from lib.metrics import bias, nmad, outlier_fraction
from lib.models.capsnet import CapsNet, logistic_to_z
from lib.utils.log import get_logger

logger = get_logger()


def predict(net: CapsNet, ds: EuclidStampsDataset, device: str,
            z_min: float, z_max: float, batch: int = 512) -> np.ndarray:
    net.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            imgs = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))])
            z_logistic, _, _ = net(imgs.to(device))
            out.append(logistic_to_z(z_logistic.cpu().numpy(), z_min, z_max))
    return np.concatenate(out)


def metrics_row(name: str, z_true: np.ndarray, z_pred: np.ndarray) -> dict[str, object]:
    return {
        "model": name,
        "n": int(len(z_true)),
        "nmad": float(nmad(z_true, z_pred)),
        "bias": float(bias(z_true, z_pred)),
        "outlier_frac": float(outlier_fraction(z_true, z_pred, threshold=0.15)),
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
    args = ap.parse_args()

    labels = pd.read_parquet(args.labels_parquet)
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

    net = CapsNet(in_channels=4, input_size=64, num_class=2, dim_capsule=16, routings=3)
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net = net.to(args.device)

    z_pred = predict(net, ds, args.device, args.z_min, args.z_max)
    z_true = ds.z

    rows = [metrics_row("capsnet_4ch", z_true, z_pred)]
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
