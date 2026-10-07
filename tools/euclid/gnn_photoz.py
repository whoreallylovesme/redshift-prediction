#!/usr/bin/env python3
"""Фото-z Euclid: трансдуктивный граф (GraphSAGE) + стек.

Воспроизводит финальный рецепт ночного перебора 2026-10-07 (gnn_lab round11):
  1) GNN на ВСЁМ каталоге (262k узлов): признаки = цвета/поток/морфология +
     zp_smart (z_pred капснета, где есть штампы, иначе PHZ) + флаги +
     собственный PHZ; граф k=16 ближайших по небу, веса рёбер - гауссианы
     по расстоянию (сигма 0.75') и разности цвета g-i (0.35); полугupervised:
     лосс только по SPE-узлам трен-тайлов, агрегация по всем узлам;
     нормировка признаков по всем узлам каталога;
  2) CatBoost-стек на узлах штампованных тайлов (34k): потоковые признаки +
     цвета + env-признаки соседей полного каталога + z_pred + PHZ соседей +
     z_gnn; веса объектов clip(exp(-|z_SPE - z_PHZ|/0.05), 0.2, 5);
     5-fold OOF по тайлам (RandomState(0)) -> медиана best_iter;
     финальный refit на train+valid с фикс. числом итераций.

Тест - тайлы 102157954 + 102160336 (как во всех экспериментах).
Эталонные метрики рецепта: NMAD 0.0882, выбросы 5.6-6.0% (стек C 0.0906/7.7%).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from scipy.spatial import cKDTree

from lib.models.capsnet import CapsNet, logistic_to_z
from lib.models.gnn import GraphSAGERegressor

D = Path("/home/kadyrov/mash/astro/datasets/euclid_q1")
RUNS = Path("/home/kadyrov/mash/astro/nn_capsnet_euclid_runs")
TEST = [102157954, 102160336]
VALID = [102159778]
FLUX = [
    "flux_u_ext_decam_unif", "flux_g_ext_decam_unif", "flux_r_ext_decam_unif",
    "flux_i_ext_decam_unif", "flux_z_ext_decam_unif", "flux_vis_unif",
    "flux_y_unif", "flux_j_unif", "flux_h_unif",
]
SIG_D, SIG_COL = 0.75, 0.35
SEED = 200


def nmad_bias_out(zp: np.ndarray, zt: np.ndarray) -> tuple[float, float, float]:
    dz = (zp - zt) / (1 + zt)
    return (
        float(1.4826 * np.median(np.abs(dz - np.median(dz)))),
        float(np.median(dz)),
        float(np.mean(np.abs(dz) > 0.15)),
    )


def colors(df: pd.DataFrame) -> dict[str, np.ndarray]:
    def col(f1: str, f2: str) -> np.ndarray:
        return -2.5 * np.log10(
            df[f1].fillna(1).clip(lower=1) / df[f2].fillna(1).clip(lower=1)
        ).to_numpy(np.float32)

    return {
        "col_gu": col("flux_g_ext_decam_unif", "flux_u_ext_decam_unif"),
        "col_ri": col("flux_r_ext_decam_unif", "flux_i_ext_decam_unif"),
        "col_gi": col("flux_g_ext_decam_unif", "flux_i_ext_decam_unif"),
        "col_rz": col("flux_r_ext_decam_unif", "flux_z_ext_decam_unif"),
        "col_vy": col("flux_vis_unif", "flux_y_unif"),
        "col_yj": col("flux_y_unif", "flux_j_unif"),
        "col_jh": col("flux_j_unif", "flux_h_unif"),
    }


def sky_xy(df: pd.DataFrame) -> np.ndarray:
    ra = np.radians(df.right_ascension.to_numpy())
    dec = np.radians(df.declination.to_numpy())
    return np.column_stack([np.degrees(ra) * np.cos(dec) * 60.0, np.degrees(dec) * 60.0])


def capsnet_zpred(
    df: pd.DataFrame, tiles: list[int], device: str, cache: Path
) -> np.ndarray:
    """Предикты CapsNet по object_id для узлов со штампами (кэш по ids)."""
    if cache.exists():
        blob = np.load(cache)
        if "object_id" in blob and len(blob["object_id"]) == len(df) and (
            blob["object_id"] == df.object_id.to_numpy()
        ).all():
            print(f"z_pred из кэша {cache}")
            return blob["zp"]
        print("кэш zp не совпадает с текущим df - пересчитываю")
    from lib.dataloaders.euclid_capsnet import EuclidStampsDataset

    net = CapsNet(in_channels=4, input_size=64, num_class=2, dim_capsule=16, routings=3)
    net.load_state_dict(
        torch.load(RUNS / "edfn_morph_v1/weights/stage_b_best.pt", map_location="cpu")
    )
    net = net.to(device).eval()
    scales = np.load(RUNS / "edfn_morph_v1/weights/norm_scales.npy").astype(np.float32)
    zp = np.full(len(df), np.nan, np.float32)
    for tile in tiles:
        ds = EuclidStampsDataset(
            stamps_dir=D / "stamps", tiles=[tile], labels=df,
            norm_scales=scales, z_min=0.0, z_max=3.0, augment=False,
        )
        if len(ds) == 0:
            continue
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(ds), 1024):
                imgs = torch.stack([ds[j][0] for j in range(i, min(i + 1024, len(ds)))])
                z_log, _, _ = net(imgs.to(device))
                out.append(logistic_to_z(z_log.cpu(), 0.0, 3.0).numpy())
        pred_by_id = dict(zip(ds.ids.tolist(), np.concatenate(out), strict=True))
        rows = df.index[df.object_id.isin(pred_by_id)].to_numpy()
        zp[rows] = [pred_by_id[int(o)] for o in df.object_id.to_numpy()[rows]]
        print(f"  капснет, тайл {tile}: {len(pred_by_id):,} предсказаний")
    np.savez(cache, object_id=df.object_id.to_numpy(), zp=zp)
    return zp


def build_graph(
    xy: np.ndarray, col_gi: np.ndarray, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist, idx = cKDTree(xy).query(xy, k=k + 1, workers=-1)
    n = xy.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].ravel()
    d, dc = dist[:, 1:].ravel(), np.abs(col_gi[src] - col_gi[dst])
    w = np.exp(-((d / SIG_D) ** 2) - ((dc / SIG_COL) ** 2)).astype(np.float32)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.from_numpy(w)
    deg = torch.zeros(n).index_add_(0, edge_index[1], torch.ones(edge_weight.size(0)))
    return edge_index, edge_weight, deg


def train_gnn(
    feats: np.ndarray,
    zt: np.ndarray,
    m_tr: np.ndarray,
    m_va: np.ndarray,
    graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str,
    epochs: int,
    patience: int,
) -> np.ndarray:
    torch.manual_seed(SEED)
    mu = np.nanmean(feats, axis=0, keepdims=True)
    sd = np.nanstd(feats, axis=0, keepdims=True) + 1e-6
    x = torch.tensor((np.nan_to_num(feats) - mu) / sd, dtype=torch.float32, device=device)
    ei, ew, deg = (t.to(device) for t in graph)
    y = torch.tensor(zt, dtype=torch.float32, device=device)
    tr, va = torch.tensor(m_tr, device=device), torch.tensor(m_va, device=device)
    model = GraphSAGERegressor(x.size(1), hidden=128, layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    lossf = torch.nn.SmoothL1Loss(beta=0.05)
    best, state, wait = 1e9, None, 0
    for _ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = lossf(model.predict_z(x, ei, ew, deg)[tr], y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = model.predict_z(x, ei, ew, deg)[va].cpu().numpy()
        n, _, _ = nmad_bias_out(pv, zt[m_va])
        if n < best - 1e-5:
            best, wait, state = n, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    assert state is not None
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        z_gnn = model.predict_z(x, ei, ew, deg).cpu().numpy()
    print(f"GNN: {_ep} эпох, лучший val NMAD {best:.4f}")
    return z_gnn.astype(np.float32)


def build_stack_frame(
    sub: pd.DataFrame,
    df_full: pd.DataFrame,
    zp_full: np.ndarray,
    z_gnn_full: np.ndarray,
    cols_full: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Кадр стека на узлах штампованных тайлов (как gnn_lab/stack_lib.py)."""
    xyf = sky_xy(df_full)
    dist_f, idx_f = cKDTree(xyf).query(xyf, k=17, workers=-1)
    nbr, ndist = idx_f[:, 1:], dist_f[:, 1:]

    def med_nb(arr: np.ndarray) -> np.ndarray:
        return np.nanmedian(np.where(np.isfinite(arr[nbr]), arr[nbr], np.nan), axis=1)

    phz_f = df_full.phz_median.to_numpy(np.float32)
    med_gi, med_rz = med_nb(cols_full["col_gi"]), med_nb(cols_full["col_rz"])
    med_vy, med_jh = med_nb(cols_full["col_vy"]), med_nb(cols_full["col_jh"])
    med_phz_nb = med_nb(phz_f)
    env = pd.DataFrame({
        "object_id": df_full.object_id.to_numpy(np.int64),
        "env_nn_dist": ndist[:, 0].astype(np.float32),
        "env_k16_dist": np.nanmedian(ndist, axis=1).astype(np.float32),
        "env_med_col_gi": med_gi.astype(np.float32),
        "env_med_col_rz": med_rz.astype(np.float32),
        "env_med_col_vy": med_vy.astype(np.float32),
        "env_med_col_jh": med_jh.astype(np.float32),
        "env_del_col_gi": (cols_full["col_gi"] - med_gi).astype(np.float32),
        "env_del_col_rz": (cols_full["col_rz"] - med_rz).astype(np.float32),
        "env_frac_redder": (cols_full["col_gi"][nbr] > cols_full["col_gi"][:, None])
            .mean(axis=1).astype(np.float32),
        "env_med_phz_nb": med_phz_nb.astype(np.float32),
        "env_del_phz": (phz_f - med_phz_nb).astype(np.float32),
    })

    oid_full = df_full.object_id.to_numpy(np.int64)
    oid_sub = pd.Series(sub.object_id.to_numpy(np.int64))
    zp_series = pd.Series(np.nan_to_num(zp_full, nan=0.4), index=oid_full)
    zg_series = pd.Series(z_gnn_full, index=oid_full)
    f = sub[["object_id"] + FLUX].copy()
    vis = sub.flux_vis_unif.fillna(1).clip(lower=1)
    for c in FLUX[1:]:
        b = c.split("_")[1]
        f[f"col_{b}"] = -2.5 * np.log10(vis / sub[c].fillna(1).clip(lower=1))
        f[f"snr_{b}"] = sub[c].fillna(0)
    f["snr_vis"] = sub.flux_vis_unif.fillna(0)
    f["z_pred"] = oid_sub.map(zp_series).to_numpy(np.float32)
    f = f.merge(env, on="object_id", how="left")
    f["z_gnn"] = oid_sub.map(zg_series).to_numpy(np.float32)
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workdir", default="/home/kadyrov/mash/astro/repo/gnn_lab/final")
    a = ap.parse_args()
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(D / "merged/edf_consensus_all_morph.parquet")
    df = df.drop_duplicates("object_id").reset_index(drop=True)
    print(f"узлов каталога: {len(df):,}")
    train_tiles = sorted(
        int(p.stem) for p in (D / "stamps").glob("*.npz")
        if p.stem.isdigit() and int(p.stem) not in TEST + VALID
    )
    tile = df.tile.to_numpy()
    m_tr, m_va, m_te = (np.isin(tile, g) for g in (train_tiles, VALID, TEST))
    print(f"трен-тайлов: {len(train_tiles)}; train {m_tr.sum():,} "
          f"valid {m_va.sum():,} test {m_te.sum():,}")

    cols = colors(df)
    phz = df.phz_median.to_numpy(np.float32)
    zp = capsnet_zpred(df, train_tiles + VALID + TEST, a.device, work / "zp_cache.npz")

    # признаки узлов GNN: базовые + zp_smart + собственный PHZ (round11)
    base = np.column_stack(
        list(cols.values())
        + [
            np.log10(df.flux_vis_unif.fillna(1).clip(lower=1)).to_numpy(np.float32),
            df.morph_class.fillna(0.5).to_numpy(np.float32),
            np.clip(df.fwhm.fillna(1.3).to_numpy(np.float32), 0, 10),
        ]
    )
    feats = np.column_stack(
        [base, np.where(np.isfinite(zp), zp, phz), np.isfinite(zp).astype(np.float32),
         np.nan_to_num(phz, nan=0.4), np.isfinite(phz).astype(np.float32)]
    )
    graph = build_graph(sky_xy(df), cols["col_gi"], k=16)
    zt = df.spe_z.to_numpy(np.float32)
    z_gnn = train_gnn(feats, zt, m_tr, m_va, graph, a.device, a.epochs, a.patience)
    gn, gb, go = nmad_bias_out(z_gnn[m_te], zt[m_te])
    print(f"GNN на тесте: NMAD={gn:.4f} bias={gb:+.4f} outliers={go:.3f}")
    np.savez(work / "z_gnn_full.npz", object_id=df.object_id.to_numpy(), z_gnn=z_gnn)

    # стек на узлах штампованных тайлов (34k), env - по полному каталогу
    sub = df[np.isin(tile, train_tiles + VALID + TEST)].reset_index(drop=True)
    X = build_stack_frame(sub, df, zp, z_gnn, cols)
    feats_cols = [c for c in X.columns if c != "object_id"]
    zt_s = sub.spe_z.to_numpy(np.float32)
    d_sp = np.abs(zt_s - sub.phz_median.to_numpy(np.float32))
    w_all = np.clip(np.exp(-d_sp / 0.05), 0.2, 5.0).astype(np.float32)
    tile_s = sub.tile.to_numpy()

    def fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, iters: int,
            va: tuple[np.ndarray, np.ndarray] | None = None) -> CatBoostRegressor:
        m = CatBoostRegressor(iterations=iters, learning_rate=0.03, depth=8,
                              loss_function="RMSE", verbose=0, random_seed=SEED,
                              early_stopping_rounds=100 if va else None)
        m.fit(x, y, sample_weight=w, eval_set=None if va is None else va,
              use_best_model=va is not None)
        return m

    tiles26 = np.array(sorted(set(tile_s[np.isin(tile_s, train_tiles)])))
    rng = np.random.RandomState(0)
    rng.shuffle(tiles26)
    folds = [set(tiles26[i::5].tolist()) for i in range(5)]
    m_tr_s = np.isin(tile_s, train_tiles)
    iters_oof: list[int] = []
    for fset in folds:
        fit_m = np.array([t in fset for t in tile_s])
        m_fit, m_oof = fit_m & m_tr_s, (~fit_m) & m_tr_s
        wf = w_all[m_fit] / w_all[m_fit].mean()
        m = fit(X.loc[m_fit, feats_cols].to_numpy(), zt_s[m_fit], wf,
                2500, va=(X.loc[m_oof, feats_cols].to_numpy(), zt_s[m_oof]))
        iters_oof.append(int(m.get_best_iteration()))
    iters = int(np.median(iters_oof)) + 1
    print(f"OOF: медиана итераций {iters} (фолды {iters_oof})")

    m_tv = np.isin(tile_s, train_tiles) | np.isin(tile_s, VALID)
    w_tv = w_all[m_tv] / w_all[m_tv].mean()
    final = fit(X.loc[m_tv, feats_cols].to_numpy(), zt_s[m_tv], w_tv, iters)
    m_te_s = np.isin(tile_s, TEST)
    zp_te = final.predict(X.loc[m_te_s, feats_cols].to_numpy())
    nmad, bias, out = nmad_bias_out(zp_te.astype(np.float32), zt_s[m_te_s])
    print(f"ФИНАЛ: NMAD={nmad:.4f} bias={bias:+.4f} outliers={out:.3f} "
          f"n={int(m_te_s.sum())} ({iters} ит., refit train+valid)")
    np.savez(work / "final_preds.npz", object_id=sub.object_id.to_numpy()[m_te_s],
             z_true=zt_s[m_te_s], z_pred=zp_te, z_gnn=z_gnn[m_te])


if __name__ == "__main__":
    main()
