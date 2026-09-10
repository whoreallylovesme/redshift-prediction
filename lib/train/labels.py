from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.isotonic import IsotonicRegression
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from lib.dataloaders.hdf5 import HDF5Dataset
from lib.models.capsnet import CapsNet, margin_loss
from lib.utils.log import get_logger
from lib.utils.utils import pick_device

logger = get_logger()


_T = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

def _train_classifier(
    model: CapsNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    decay_rate: float,
    lam_recon: float,
    num_pixels: int,
) -> None:
    opt = Adam(model.parameters(), lr=lr)
    sch = ExponentialLR(opt, gamma=decay_rate)
    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        for images, _z, morph in train_loader:
            images = images.to(device, non_blocking=True)
            morph = morph.to(device, non_blocking=True)
            _, probs, recon = model(images, target_class=morph.long())
            onehot = F.one_hot(morph.long(), num_classes=model.num_class).float()
            loss = margin_loss(probs, onehot) + lam_recon * num_pixels * F.mse_loss(recon, images)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, _z, morph in val_loader:
                images = images.to(device, non_blocking=True)
                morph = morph.to(device, non_blocking=True)
                _, probs, _ = model(images, target_class=None)
                correct += int((probs.argmax(dim=1) == morph.long()).sum())
                total += int(morph.shape[0])
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        sch.step()
        logger.info(
            f"    epoch {epoch + 1}/{epochs}: val accuracy = {acc:.4f} "
            f"(best {best_acc:.4f}) ({time.time() - t0:.0f}s)",
        )
    if best_state is not None:
        model.load_state_dict(best_state)

@torch.no_grad()
def _predict_probs(
    model: CapsNet,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for images, _z, _morph in loader:
        images = images.to(device, non_blocking=True)
        _, probs, _ = model(images, target_class=None)
        out.append(probs[:, 1].cpu().numpy())
    return np.concatenate(out)

def run_generate(config_path: Path) -> None:
    raw = yaml.safe_load(config_path.read_text())
    cfg: dict[str, Any] = raw.get("labels", {})
    iters = int(cfg.get("iters", 3))
    epochs_per_iter = int(cfg.get("epochs_per_iter", 5))
    fine_lr_raw = cfg.get("fine_lr")
    fine_lr = float(fine_lr_raw) if fine_lr_raw is not None else None
    threshold = float(cfg.get("threshold", 0.8))
    max_samples_raw = cfg.get("max_samples")
    max_samples = int(max_samples_raw) if max_samples_raw is not None else None
    seed_raw = cfg.get("seed_parquet")
    init_raw = cfg.get("init_weights")
    out_raw = cfg.get("out")

    h5_path = Path(raw["h5_path"])
    seed_parquet = Path(seed_raw) if seed_raw else Path(raw["morph_parquet"])
    init_weights = Path(init_raw) if init_raw else None
    out_parquet = (
        Path(out_raw) if out_raw else h5_path.parent / "hayat_train_morph_generated.parquet"
    )

    device = pick_device(raw.get("device", "auto"))
    logger.info(f"[labels] device={device}")

    seed = pd.read_parquet(seed_parquet)
    with h5py.File(h5_path, "r") as f:
        n = f["images"].shape[0]
        objids = [int(x) for x in f["ObjID"][:].astype(str)]
    if len(seed) == n:
        labels = seed["gz_morpho_binary"].to_numpy(dtype=np.float32).copy()
        cal_val_full = (
            seed["cal_val"].to_numpy() if "cal_val" in seed.columns
            else np.zeros(n, dtype=bool)
        )
    else:
        lab_map = dict(zip(seed["objid"].tolist(), seed["gz_morpho_binary"].tolist(),
                           strict=True))
        cv_map = (
            dict(zip(seed["objid"].tolist(), seed["cal_val"].tolist(), strict=True))
            if "cal_val" in seed.columns else {}
        )
        labels = np.array([lab_map.get(o, np.nan) for o in objids], dtype=np.float32)
        cal_val_full = np.array([bool(cv_map.get(o, False)) for o in objids], dtype=bool)
        logger.info(
            f"[labels] seed parquet aligned by objid: {len(seed)} rows → "
            f"{int((~np.isnan(labels)).sum())} labels of {n}",
        )
    indices = np.arange(n)
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(raw.get("seed", 200))
        indices = rng.choice(n, size=max_samples, replace=False)
        indices.sort()
        labels = labels[indices]
        cal_val_full = cal_val_full[indices]
        objids = [objids[i] for i in indices]
        n = len(indices)

    crop = int(raw.get("crop_size", 64))
    scale = float(raw.get("image_scale", 10.0))
    num_pixels = int(raw.get("in_channels", 5)) * crop * crop
    lam_recon = float(raw.get("lam_recon", 0.005))

    torch.manual_seed(int(raw.get("seed", 200)))
    np.random.seed(int(raw.get("seed", 200)))
    model = CapsNet(
        in_channels=int(raw.get("in_channels", 5)), input_size=crop,
        num_class=2, dim_capsule=16, routings=3,
    ).to(device)
    if init_weights is not None:
        model.load_state_dict(
            torch.load(init_weights, map_location=device, weights_only=True)
        )
        logger.info(f"[labels] warm start from {init_weights}")

    labeled_pos = np.where(~np.isnan(labels))[0]
    if bool(cal_val_full.any()):
        val_pos = set(np.where(cal_val_full)[0].tolist())
        logger.info(f"[labels] reusing persisted calibration split: {len(val_pos)} objects")
    else:
        rng = np.random.default_rng(raw.get("seed", 200))
        perm = rng.permutation(labeled_pos)
        val_pos = set(perm[: max(int(0.15 * len(perm)), 1)].tolist())

    def loader_for(pos: np.ndarray, augment: bool) -> DataLoader:
        lab = labels[pos].astype(np.float32)
        ds = HDF5Dataset(
            h5_path, indices[pos], lab, lab,
            image_scale=scale, crop_size=crop, augment=augment,
        )
        return DataLoader(ds, batch_size=int(raw.get("batch_size", 256)),
                          shuffle=augment, num_workers=3, drop_last=augment)

    for it in range(1, iters + 1):
        labeled = np.where(~np.isnan(labels))[0]
        train_pos = np.array([p for p in labeled if int(p) not in val_pos])
        valpos_arr = np.array(sorted(val_pos))
        logger.info(
            f"[labels] iter {it}/{iters}: labeled={len(labeled)} "
            f"(train={len(train_pos)}, val={len(valpos_arr)})",
        )

        _train_classifier(
            model, loader_for(train_pos, True), loader_for(valpos_arr, False),
            device, epochs_per_iter,
            fine_lr if fine_lr is not None else float(raw.get("lr", 5e-4)),
            float(raw.get("decay_rate", 0.95)), lam_recon, num_pixels,
        )

        probs_all = _predict_probs(model, loader_for(np.arange(n), False), device)

        val_idx = valpos_arr
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(probs_all[val_idx], labels[val_idx])
        cal = iso.predict(probs_all).astype(np.float32)

        unl = np.where(np.isnan(labels))[0]
        new_sp = unl[(cal[unl] > threshold)]
        new_el = unl[(cal[unl] < 1.0 - threshold)]
        labels[new_sp] = 1.0
        labels[new_el] = 0.0
        cov = 100.0 * float((~np.isnan(labels)).sum()) / n
        logger.info(
            f"[labels] iter {it}: +{len(new_sp)} spiral, +{len(new_el)} elliptical "
            f"→ coverage {cov:.1f}%",
        )

        val_acc = float(
            np.mean((iso.predict(probs_all[val_idx]) > 0.5) == (labels[val_idx] == 1.0))
        )
        logger.info(f"[labels] iter {it}: calibrated val accuracy {val_acc:.4f}")

        out = pd.DataFrame({
            "objid": objids,
            "gz_morpho_binary": labels,
            "cal_val": np.isin(np.arange(n), valpos_arr),
        })
        out.to_parquet(out_parquet, index=False)
        torch.save(model.state_dict(), out_parquet.with_suffix(".pt"))
        logger.info(f"[labels] iter {it}: checkpoint saved → {out_parquet} (+ .pt)")

    n_lab = int((~np.isnan(labels)).sum())
    logger.info(f"[labels] DONE: {n_lab}/{n} labeled ({100.0 * n_lab / n:.1f}%) → {out_parquet}")

