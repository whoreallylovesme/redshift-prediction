from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import h5py
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from lib.config import FeaturesConfig, PytorchTrainConfig, SklearnTrainConfig
from lib.dataloaders.common import subsample_indices
from lib.dataloaders.dataset import read_dataset
from lib.dataloaders.hdf5 import HDF5Dataset
from lib.dataloaders.teddy_capsnet import TeddyCapsnetDataset, _scan
from lib.metrics import evaluate_regression
from lib.utils.log import get_logger
from lib.utils.registry import TORCH_PREDICTORS, register_torch_predictor
from lib.utils.utils import load_yaml, load_yaml_as, pick_device

logger = get_logger()


logger = get_logger()


def build_predictor(raw: dict[str, Any], weights: Path) -> Any:
    from lib.models.predictors import CapsNetPredictor

    device = pick_device(str(raw.get("device", "auto")))
    return CapsNetPredictor(
        weights=str(weights),
        in_channels=int(raw.get("in_channels", 4)),
        input_size=int(raw.get("crop_size", 64)),
        z_min=float(raw.get("z_min", 0.0)),
        z_max=float(raw.get("z_max", 0.4)),
        device=device,
    )


def _predict(
    predictor: Any, loader: DataLoader,
) -> dict[str, np.ndarray]:
    outs: dict[str, list[np.ndarray]] = {}
    for images, _z, _morph in loader:
        res = predictor.predict(images)
        for k, v in res.items():
            outs.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}

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
    ds = HDF5Dataset(
        h5, idx, z_spec[idx], nan_morph,
        image_scale=float(raw.get("image_scale", 10.0)),
        crop_size=int(raw.get("crop_size", 64)), augment=False,
    )
    loader = DataLoader(ds, batch_size=int(raw.get("batch_size", 256)), num_workers=3)
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
    loader = DataLoader(ds, batch_size=int(raw.get("batch_size", 256)), num_workers=3)
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


def _inference_pytorch(
    model_path: str, cfg: PytorchTrainConfig, data_dir: str, catalog: str | None, out: str,
) -> None:
    import torch

    from lib.dataloaders.npz import compute_stats, load_npz, parse_catalog

    scope: dict[str, object] = {}
    exec(cfg.estimator, scope)
    estimator = scope["estimator"]
    if not isinstance(estimator, torch.nn.Module):
        raise TypeError("Loaded object is not a torch.nn.Module")
    model = estimator

    device = pick_device("auto")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    paths = sorted(Path(data_dir).glob("*.npz"))
    if not paths:
        raise ValueError(f"No .npz files in {data_dir}")

    norm_dir = Path(cfg.norm_from or data_dir)
    norm_paths = sorted(norm_dir.glob("*.npz"))
    mean, std = compute_stats(norm_paths)

    z_true_map: dict[str, float] = {}
    if catalog is not None:
        z_true_map = parse_catalog(catalog)

    from tqdm import tqdm

    crop_size = cfg.crop_size
    out_size = cfg.out_size
    batch_size = cfg.batch_size
    results: list[dict[str, object]] = []
    skipped_files: list[tuple[str, str]] = []
    n_batches = (len(paths) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(paths), batch_size), total=n_batches, desc="Inference"):
        batch_paths = paths[i : i + batch_size]
        loaded_paths: list[Path] = []
        batch_tensors = []
        for fp in batch_paths:
            try:
                batch_tensors.append(
                    load_npz(fp, mean, std, crop_size=crop_size, out_size=out_size)
                )
                loaded_paths.append(fp)
            except Exception as exc:
                skipped_files.append((fp.name, f"{type(exc).__name__}: {exc}"))
        if not batch_tensors:
            continue

        batch = torch.cat(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            model_out = model(batch)
            pred_t = model_out[0] if isinstance(model_out, tuple) else model_out
            preds = pred_t.squeeze(-1).cpu().numpy()
        if preds.ndim == 0:
            preds = preds.reshape(1)

        for fp, pred in zip(loaded_paths, preds, strict=True):
            row: dict[str, object] = {"y_pred": float(pred)}
            if fp.stem in z_true_map:
                row["y_true"] = z_true_map[fp.stem]
            results.append(row)

    df = pd.DataFrame(results)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    if skipped_files:
        logger.info(f"Skipped {len(skipped_files)} corrupted files during inference.")
        for filename, reason in skipped_files[:10]:
            logger.info(f"  {filename}: {reason}")

def _inference_sklearn(
    model_path: str, test_path: str, features_path: str, feature_set: str, out: str,
) -> None:
    fcfg = load_yaml_as(features_path, FeaturesConfig)
    target = fcfg.target
    feature_list = fcfg.feature_sets[feature_set]

    df = read_dataset(test_path)
    X = df[feature_list]
    y = df[target]

    m = joblib.load(model_path)
    pred = m.predict(X)

    preds = pd.DataFrame({
        "y_true": y.to_numpy(),
        "y_pred": pred,
    })
    preds.to_parquet(out, index=False)

def _infer_capsnet(config_path: str, **_: object) -> None:
    run_inference(Path(config_path))

def _infer_pytorch_default(
    config_path: str,
    model_path: str | None = None,
    out: str | None = None,
    data_dir: str | None = None,
    catalog: str | None = None,
    **_: object,
) -> None:
    cfg = PytorchTrainConfig.model_validate(load_yaml(config_path))
    out_dir = cfg.out_dir or f"nn_runs/{Path(config_path).stem}"
    model_path = model_path or f"{out_dir}/best_model.pt"
    out = out or f"{out_dir}/preds.parquet"
    data_dir = data_dir or cfg.data_dir
    catalog = catalog if catalog is not None else cfg.catalog
    _inference_pytorch(
        cast(str, model_path), cfg, cast(str, data_dir), catalog, out,
    )

def _infer_sklearn_entry(
    config_path: str,
    model_path: str | None = None,
    out: str | None = None,
    test_path: str | None = None,
    features_path: str | None = None,
    feature_set: str | None = None,
    **_: object,
) -> None:
    scfg = load_yaml_as(config_path, SklearnTrainConfig)
    out_dir = scfg.out_dir or f"runs/{Path(config_path).stem}"
    model_path = model_path or f"{out_dir}/model.pkl"
    out = out or f"{out_dir}/preds.parquet"
    test_path = test_path or scfg.test_path
    features_path = features_path or scfg.features_path
    feature_set = feature_set or scfg.feature_set
    _inference_sklearn(
        cast(str, model_path),
        cast(str, test_path),
        cast(str, features_path),
        cast(str, feature_set),
        out,
    )

register_torch_predictor("capsnet", _infer_capsnet)

def inference(
    model_path: str | None = None,
    out: str | None = None,
    config_path: str | None = None,
    test_path: str | None = None,
    features_path: str | None = None,
    feature_set: str | None = None,
    data_dir: str | None = None,
    catalog: str | None = None,
) -> None:
    if config_path is None:
        raise ValueError("config_path is required")
    raw_cfg = load_yaml(config_path)
    model_type = raw_cfg.get("model_type", "sklearn") if isinstance(raw_cfg, dict) else "sklearn"
    if model_type == "lightning":
        model = str(raw_cfg.get("model", "")) if isinstance(raw_cfg, dict) else ""
        predictor: Callable[..., object] = TORCH_PREDICTORS.get(
            model, _infer_pytorch_default,
        )
    elif model_type == "sklearn":
        predictor = _infer_sklearn_entry
    else:
        raise ValueError(f"unsupported model_type {model_type!r}")
    predictor(
        config_path, model_path=model_path, out=out, test_path=test_path,
        features_path=features_path, feature_set=feature_set,
        data_dir=data_dir, catalog=catalog,
    )
