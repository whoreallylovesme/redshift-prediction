#!/usr/bin/env python3
"""Стекинг: capsnet-предикты (картинки) + фотометрия (9 полос) -> CatBoost.

Схема бьёт SOM информационно: и пиксели, и цвета. Тест — те же тайлы EDF-N.

Вызов: python tools/euclid/stack_photoz.py CHECKPOINT [LABELS_PARQUET]
(пути/гиперпараметры переопределяются флагами, см. --help).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor

from lib.dataloaders.euclid_capsnet import EuclidStampsDataset
from lib.models.capsnet import CapsNet, logistic_to_z
from lib.utils.log import get_logger

logger = get_logger()

DEFAULT_DATA_DIR = Path("/home/kadyrov/mash/astro/datasets/euclid_q1")
DEFAULT_OUT_DIR = Path("/home/kadyrov/mash/astro/nn_capsnet_euclid_runs/stack")
TEST = [102157954, 102160336]
VALID = [102159778]

FLUX = ["flux_u_ext_decam_unif", "flux_g_ext_decam_unif", "flux_r_ext_decam_unif",
        "flux_i_ext_decam_unif", "flux_z_ext_decam_unif", "flux_vis_unif",
        "flux_y_unif", "flux_j_unif", "flux_h_unif"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="чекпоинт CapsNet (stage_b)")
    ap.add_argument("labels_parquet", nargs="?", default=None,
                    help="по умолчанию <data-dir>/merged/edfn_labels_consensus.parquet")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--test-tiles", type=int, nargs="+", default=TEST)
    ap.add_argument("--valid-tiles", type=int, nargs="+", default=VALID)
    ap.add_argument("--z-min", type=float, default=0.0)
    ap.add_argument("--z-max", type=float, default=3.0)
    ap.add_argument("--in-channels", type=int, default=4)
    ap.add_argument("--input-size", type=int, default=64)
    ap.add_argument("--num-class", type=int, default=2)
    ap.add_argument("--dim-capsule", type=int, default=16)
    ap.add_argument("--routings", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=2500)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)
    ap.add_argument("--random-seed", type=int, default=200)
    return ap.parse_args()


def train_tiles(stamps_dir: Path, excluded: list[int]) -> list[int]:
    return sorted(int(f.stem) for f in stamps_dir.glob("*.npz") if int(f.stem) not in excluded)


def predict_tiles(net: CapsNet, stamps_dir: Path, tiles: list[int], labels: pd.DataFrame,
                  scales: np.ndarray, z_min: float = 0.0, z_max: float = 3.0,
                  device: str = "cuda", batch: int = 1024) -> pd.DataFrame:
    ds = EuclidStampsDataset(
        stamps_dir=stamps_dir, tiles=tiles, labels=labels,
        norm_scales=scales, z_min=z_min, z_max=z_max, augment=False,
    )
    net.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            imgs = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))])
            z_log, _, _ = net(imgs.to(device))
            out.append(logistic_to_z(z_log.cpu(), z_min, z_max).numpy())
    return pd.DataFrame({"object_id": ds.ids, "z_pred": np.concatenate(out)})


def feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vis = df["flux_vis_unif"].clip(lower=1)
    for c in FLUX[1:]:
        b = c.split("_")[1]
        out[f"col_{b}"] = -2.5 * np.log10(vis / df[c].clip(lower=1))
        err = df[c.replace("flux_", "fluxerr_")].clip(lower=1)
        out[f"snr_{b}"] = df[c] / err
    out["snr_vis"] = df["flux_vis_unif"] / df["fluxerr_vis_unif"].clip(lower=1)
    return out


def build_capsnet(a: argparse.Namespace) -> CapsNet:
    net = CapsNet(in_channels=a.in_channels, input_size=a.input_size, num_class=a.num_class,
                  dim_capsule=a.dim_capsule, routings=a.routings)
    net.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    return net.to("cuda")


def fit_catboost(a: argparse.Namespace, ftr: pd.DataFrame, fva: pd.DataFrame,
                 cols: list[str]) -> CatBoostRegressor:
    model = CatBoostRegressor(iterations=a.iterations, learning_rate=a.learning_rate,
                              depth=a.depth, loss_function="RMSE",
                              early_stopping_rounds=a.early_stopping_rounds,
                              verbose=500, random_seed=a.random_seed)
    model.fit(ftr[cols], ftr["spe_z"], eval_set=(fva[cols], fva["spe_z"]), use_best_model=True)
    return model


def report_metrics(name: str, zt: np.ndarray, zp: np.ndarray) -> None:
    dz = (zp - zt) / (1 + zt)
    nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    logger.info(f"{name}: n={len(zt):,} NMAD={nmad:.4f} "
                f"bias={np.median(dz):+.4f} outliers={np.mean(np.abs(dz) > 0.15):.3f}")


def main() -> None:
    a = parse_args()
    labels_path = a.labels_parquet or a.data_dir / "merged" / "edfn_labels_consensus.parquet"
    labels = pd.read_parquet(labels_path)
    net = build_capsnet(a)
    scales = np.load(Path(a.checkpoint).parent / "norm_scales.npy")

    stamps = a.data_dir / "stamps"
    train = train_tiles(stamps, a.test_tiles + a.valid_tiles)
    tr_pred = predict_tiles(net, stamps, train, labels, scales, a.z_min, a.z_max)
    te_pred = predict_tiles(net, stamps, a.test_tiles, labels, scales, a.z_min, a.z_max)
    va_pred = predict_tiles(net, stamps, a.valid_tiles, labels, scales, a.z_min, a.z_max)

    tr = labels[labels.tile.isin(train)].merge(tr_pred, on="object_id")
    va = labels[labels.tile.isin(a.valid_tiles)].merge(va_pred, on="object_id")
    te = labels[labels.tile.isin(a.test_tiles)].merge(te_pred, on="object_id")

    ftr, fva, fte = feats(tr), feats(va), feats(te)
    cols = FLUX + [c for c in ftr.columns if c.startswith(("col_", "snr_"))] + ["z_pred"]

    model = fit_catboost(a, ftr, fva, cols)

    zt = te["spe_z"].to_numpy()
    for name, zp in [("stack_catboost+capsnet", model.predict(fte[cols])),
                     ("capsnet_only", te["z_pred"].to_numpy())]:
        report_metrics(name, zt, zp)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(a.out_dir / "stack.cbm"))


if __name__ == "__main__":
    main()
