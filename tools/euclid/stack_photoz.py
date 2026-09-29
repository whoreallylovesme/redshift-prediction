#!/usr/bin/env python3
"""Стекинг: capsnet-предикты (картинки) + фотометрия (9 полос) -> CatBoost.

Схема бьёт SOM информационно: и пиксели, и цвета. Тест — те же тайлы EDF-N.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from catboost import CatBoostRegressor
from lib.dataloaders.euclid_capsnet import EuclidStampsDataset
from lib.models.capsnet import CapsNet, logistic_to_z

D = "/home/kadyrov/mash/astro/datasets/euclid_q1"
TRAIN = [102158895, 102159196, 102159195, 102158896, 102158894, 102160057, 102158893, 102159779]
VALID = [102159778]
TEST = [102157954, 102160336]

FLUX = ["flux_u_ext_decam_unif", "flux_g_ext_decam_unif", "flux_r_ext_decam_unif",
        "flux_i_ext_decam_unif", "flux_z_ext_decam_unif", "flux_vis_unif",
        "flux_y_unif", "flux_j_unif", "flux_h_unif"]


def predict_tiles(net: CapsNet, tiles: list[int], labels: pd.DataFrame, scales: np.ndarray,
                  device: str = "cuda") -> pd.DataFrame:
    ds = EuclidStampsDataset(
        stamps_dir=Path(f"{D}/stamps"), tiles=tiles, labels=labels,
        norm_scales=scales, z_min=0.0, z_max=3.0, augment=False,
    )
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 1024):
            imgs = torch.stack([ds[j][0] for j in range(i, min(i + 1024, len(ds)))])
            z_log, _, _ = net(imgs.to(device))
            out.append(logistic_to_z(z_log.cpu(), 0.0, 3.0).numpy())
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


def main() -> None:
    ckpt = sys.argv[1]
    labels = pd.read_parquet(f"{D}/merged/edfn_labels_consensus.parquet")
    net = CapsNet(in_channels=4, input_size=64, num_class=2, dim_capsule=16, routings=3)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net = net.to("cuda")
    scales = np.load(Path(ckpt).parent / "norm_scales.npy")

    tr_pred = predict_tiles(net, TRAIN, labels, scales)
    te_pred = predict_tiles(net, TEST, labels, scales)
    va_pred = predict_tiles(net, VALID, labels, scales)

    tr = labels[labels.tile.isin(TRAIN)].merge(tr_pred, on="object_id")
    va = labels[labels.tile.isin(VALID)].merge(va_pred, on="object_id")
    te = labels[labels.tile.isin(TEST)].merge(te_pred, on="object_id")

    ftr, fva, fte = feats(tr), feats(va), feats(te)
    cols = FLUX + [c for c in ftr.columns if c.startswith(("col_", "snr_"))] + ["z_pred"]

    model = CatBoostRegressor(iterations=2500, learning_rate=0.03, depth=8,
                              loss_function="RMSE", early_stopping_rounds=100,
                              verbose=500, random_seed=200)
    model.fit(ftr[cols], ftr["spe_z"], eval_set=(fva[cols], fva["spe_z"]), use_best_model=True)

    zt = te["spe_z"].to_numpy()
    for name, zp in [("stack_catboost+capsnet", model.predict(fte[cols])),
                     ("capsnet_only", te["z_pred"].to_numpy())]:
        dz = (zp - zt) / (1 + zt)
        print(f"{name}: n={len(zt):,} NMAD={1.4826*np.median(np.abs(dz-np.median(dz))):.4f} "
              f"bias={np.median(dz):+.4f} outliers={np.mean(np.abs(dz)>0.15):.3f}")

    out = Path("/home/kadyrov/mash/astro/nn_capsnet_euclid_runs/stack")
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "stack.cbm"))


if __name__ == "__main__":
    main()
