from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from hydra.utils import instantiate
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from lib.train.lightning import predict_batches
from lib.utils.log import get_logger
from lib.utils.utils import pick_device
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


def _predict(predictor: Any, loader: DataLoader) -> dict[str, np.ndarray]:
    results = predict_batches(predictor, loader, predictor.device)
    if not results:
        return {"z_pred": np.empty(0), "p_spiral": np.empty(0)}
    return {
        key: np.concatenate([batch[key].cpu().numpy() for batch in results])
        for key in ("z_pred", "p_spiral")
    }


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


def inference_data(config: Any) -> tuple[pd.DataFrame, DataLoader]:
    if not isinstance(config, dict) or "_target_" not in config:
        raise ValueError("CapsNet inference requires data._target_")
    result = instantiate(config)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("CapsNet inference data factory must return (DataFrame, DataLoader)")
    frame, loader = result
    if not isinstance(frame, pd.DataFrame) or not isinstance(loader, DataLoader):
        raise TypeError("CapsNet inference data factory must return (DataFrame, DataLoader)")
    return frame, loader


def run_inference(config_path: Path) -> None:
    raw = yaml.safe_load(config_path.read_text())
    inf: dict[str, Any] = raw.get("inference", {})
    weights = Path(inf.get("weights", Path(raw["out_dir"]) / "weights" / "stage_b_best.pt"))
    predictor = build_predictor(raw, weights)
    logger.info(f"[infer] weights={weights}")

    targets: list[dict[str, Any]] = inf.get("splits") or [inf]
    for target in targets:
        frame, loader = inference_data(target.get("data"))
        result = _predict(predictor, loader)
        if len(frame) != len(result["z_pred"]) or len(frame) != len(result["p_spiral"]):
            raise ValueError("CapsNet predictions do not align with inference data")
        df = frame.copy()
        df["z_pred"] = result["z_pred"]
        df["p_spiral"] = result["p_spiral"]
        out = Path(target["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        _report(str(out), df["z_spec"].to_numpy(), df["z_pred"].to_numpy())
        if inf.get("compare_npz"):
            _compare_with_paper(df, Path(str(inf["compare_npz"])))
