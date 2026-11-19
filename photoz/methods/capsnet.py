from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from lib.dataloaders.common import subsample_indices
from lib.train.lightning import predict_batches
from lib.utils.log import get_logger
from lib.utils.utils import pick_device
from photoz.data.hayat import HayatDataset
from photoz.data.teddy_capsnet import TeddyCapsnetDataset, _scan
from photoz.metrics import evaluate_regression

logger = get_logger()


def build_predictor(raw: dict[str, Any], weights: Path) -> Any:
    from photoz.models.nn.predictors import CapsNetPredictor
    from photoz.training.capsnet_lightning import TrainConfig

    device = pick_device(str(raw.get("device", "auto")))
    return CapsNetPredictor(
        weights=str(weights),
        in_channels=int(raw.get("in_channels", TrainConfig.in_channels)),
        input_size=int(raw.get("crop_size", 64)),
        z_min=float(raw.get("z_min", 0.0)),
        z_max=float(raw.get("z_max", 0.4)),
        device=device,
    )


def _predict(
    predictor: Any, loader: DataLoader,
) -> dict[str, np.ndarray]:
    results = predict_batches(predictor, loader, predictor.device)
    if not results:
        return {"z_pred": np.empty(0), "p_spiral": np.empty(0)}
    return {key: np.concatenate([batch[key].cpu().numpy() for batch in results])
            for key in ("z_pred", "p_spiral")}


def _report(tag: str, z_true: np.ndarray, z_pred: np.ndarray) -> None:
    m = evaluate_regression(pd.Series(z_true), z_pred)
    logger.info(
        f"[infer] {tag}: N={len(z_true)} nmad={m['sigma_nmad']:.4f} "
        f"bias={m['bias']:+.4f} outlier={m['f_outlier_percent']:.2f}%",
    )

def _compare_with_paper(df: pd.DataFrame, compare_npz: Path) -> None:
    theirs = np.load(compare_npz)
    t_id = theirs["test_id"].astype(np.int64)
    t_z = theirs["z_phot"]
    t_map = dict(zip(t_id.tolist(), t_z.tolist(), strict=True))
    key = "specobjid" if "specobjid" in df.columns else "objid"
    shared = df[df[key].isin(t_map.keys())].copy()
    shared["z_paper"] = shared[key].map(t_map).astype(float)
    logger.info(
        f"[infer] comparison with paper predictions: {len(shared)} shared galaxies "
        f"(join key: {key})",
    )
    if len(shared) > 100:
        _report("OURS  on shared", shared["z_spec"].to_numpy(), shared["z_pred"].to_numpy())
        _report("PAPER on shared", shared["z_spec"].to_numpy(), shared["z_paper"].to_numpy())
        pe = pearsonr(shared["z_pred"].to_numpy(), shared["z_paper"].to_numpy())
        sp = spearmanr(shared["z_pred"].to_numpy(), shared["z_paper"].to_numpy())
        logger.info(
            f"[infer] corr(ours, paper): pearson={pe.statistic:.4f} "
            f"spearman={sp.statistic:.4f}",
        )

def _infer_h5(
    raw: dict[str, Any], inf: dict[str, Any], predictor: Any,
) -> pd.DataFrame:
    h5 = Path(inf.get("h5_path", raw["h5_path"]))
    max_samples = inf.get("max_samples")
    with h5py.File(h5, "r") as f:
        n = f["images"].shape[0]
        objids = np.array([int(x) for x in f["ObjID"][:].astype(str)], dtype=np.int64)
        specids = np.array([int(x) for x in f["specObjID"][:].astype(str)], dtype=np.int64)
        z_spec = f["specz_redshift"][:].astype(np.float32)
    idx = np.arange(n)
    if max_samples is not None and n > int(max_samples):
        idx = subsample_indices(n, int(max_samples), int(raw.get("seed", 200)))
    nan_morph = np.full(len(idx), np.nan, dtype=np.float32)
    ds = HayatDataset(
        h5, idx, z_spec[idx], nan_morph,
        image_scale=float(raw.get("image_scale", 10.0)),
        crop_size=int(raw.get("crop_size", 64)), augment=False,
    )
    loader = DataLoader(
        ds, batch_size=int(raw.get("batch_size", 256)),
        num_workers=int(inf.get("num_workers", raw.get("num_workers", 3))),
    )
    res = _predict(predictor, loader)
    return pd.DataFrame({
        "objid": objids[idx], "specobjid": specids[idx], "z_spec": z_spec[idx],
        "z_pred": res["z_pred"], "p_spiral": res["p_spiral"],
    })

def _infer_split(
    raw: dict[str, Any], inf: dict[str, Any], split: dict[str, str], predictor: Any,
) -> pd.DataFrame:
    images_dir = Path(split["images_dir"])
    catalog_path = Path(split["catalog"])
    gz1 = Path(split.get("gz1_parquet", raw["gz1_parquet"]))
    w1w2_dir = Path(split["w1w2_dir"]) if split.get("w1w2_dir") else None
    max_samples = inf.get("max_samples")
    files, z_all, morph_all = _scan(images_dir, catalog_path, gz1, w1w2_dir=w1w2_dir)
    if max_samples is not None and len(files) > int(max_samples):
        k = subsample_indices(len(files), int(max_samples), int(raw.get("seed", 200)))
        files = [files[i] for i in k]
        z_all, morph_all = z_all[k], morph_all[k]
    ds = TeddyCapsnetDataset(
        files, z_all, morph_all,
        in_channels=int(raw.get("in_channels", 4)),
        crop_size=int(raw.get("crop_size", 64)),
        image_scale=float(raw.get("image_scale", 10.0)), augment=False,
        per_channel_norm=bool(raw.get("per_channel_norm", False)),
        w1w2_dir=w1w2_dir,
    )
    loader = DataLoader(
        ds, batch_size=int(raw.get("batch_size", 256)),
        num_workers=int(inf.get("num_workers", raw.get("num_workers", 3))),
    )
    res = _predict(predictor, loader)
    return pd.DataFrame({
        "objid": [int(f.stem) for f in files], "z_spec": z_all,
        "z_pred": res["z_pred"], "p_spiral": res["p_spiral"],
    })

def run_inference(config_path: Path) -> None:
    raw = yaml.safe_load(config_path.read_text())
    inf: dict[str, Any] = raw.get("inference", {})
    weights = Path(inf.get("weights", Path(raw["out_dir"]) / "weights" / "stage_b_best.pt"))
    predictor = build_predictor(raw, weights)
    logger.info(f"[infer] weights={weights} dataset={raw.get('dataset', 'hayat')}")

    targets: list[tuple[str, pd.DataFrame]]
    if "splits" in inf:
        targets = [
            (str(sp["out"]), _infer_split(raw, inf, sp, predictor)) for sp in inf["splits"]
        ]
    else:
        targets = [(str(inf["out"]), _infer_h5(raw, inf, predictor))]

    for out_path, df in targets:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        _report(str(out), df["z_spec"].to_numpy(), df["z_pred"].to_numpy())
        if inf.get("compare_npz"):
            _compare_with_paper(df, Path(str(inf["compare_npz"])))
