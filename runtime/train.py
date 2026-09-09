import json
import time
from pathlib import Path
from typing import cast

import click
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from lib.config_models import FeaturesConfig, PytorchTrainConfig, SklearnTrainConfig
from lib.dataset import read_dataset
from lib.estimator_loader import load_estimator
from lib.log import get_logger
from lib.training import (
    evaluate,
    feature_selection,
    hyperparameter_optimization,
    train_model,
)
from lib.utils import load_yaml, load_yaml_as

logger = get_logger()



def _train_pytorch(cfg: PytorchTrainConfig, outdir: str) -> None:
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from tqdm import tqdm

    from lib.metrics import evaluate_regression
    from lib.nn.dataloader_photoz import get_photoz_dataloaders
    from lib.nn.train_utils import WeightedHuberLoss, unfreeze_backbone

    # Load model from estimator code
    scope: dict[str, object] = {}
    exec(cfg.estimator, scope)
    estimator = scope["estimator"]
    if not isinstance(estimator, nn.Module):
        raise TypeError("Loaded object is not a torch.nn.Module")
    model = estimator

    epochs = cfg.epochs
    batch_size = cfg.batch_size
    lr = cfg.lr
    weight_decay = cfg.weight_decay
    freeze_epochs = cfg.freeze_epochs
    seed = cfg.seed
    num_workers = cfg.num_workers
    weighted_sampling = cfg.weighted_sampling
    z_weight_alpha = cfg.z_weight_alpha
    label_noise = cfg.label_noise
    head_prefix = cfg.head_prefix

    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    max_samples = cfg.max_samples
    crop_size = cfg.crop_size
    physics_augment = cfg.physics_augment

    train_loader, val_loader = get_photoz_dataloaders(
        cfg.data_dir, cfg.catalog,
        batch_size=batch_size, num_workers=num_workers,
        train_ratio=cfg.train_ratio, seed=seed,
        weighted_sampling=weighted_sampling,
        max_samples=max_samples,
        crop_size=crop_size,
        physics_augment=physics_augment,
    )

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total:,} total, {trainable:,} trainable")

    if z_weight_alpha > 0:
        criterion: nn.Module = WeightedHuberLoss(delta=0.1, alpha=z_weight_alpha)
    else:
        criterion = nn.HuberLoss(delta=0.1)

    head_params = [p for n, p in model.named_parameters() if n.startswith(head_prefix)]
    optimizer = AdamW(head_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    best_nmad = float("inf")

    for ep in range(1, epochs + 1):
        t_ep = time.time()

        if ep == freeze_epochs + 1 and freeze_epochs > 0:
            unfreeze_backbone(model)
            logger.info("\n>>> Backbone unfrozen!")
            optimizer = AdamW([
                {"params": [p for n, p in model.named_parameters()
                            if n.startswith(head_prefix)], "lr": lr},
                {"params": [p for n, p in model.named_parameters()
                            if not n.startswith(head_prefix)], "lr": lr / 10},
            ], weight_decay=weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - ep + 1)

        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            if label_noise > 0:
                y = y + torch.randn_like(y) * label_noise
            optimizer.zero_grad()
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        scheduler.step()

        # Eval
        model.eval()
        all_pred: list[np.ndarray] = []
        all_true: list[np.ndarray] = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                pred = model(x).squeeze(-1).cpu().numpy()
                all_pred.append(pred)
                all_true.append(y.numpy())
        y_pred = np.concatenate(all_pred)
        y_true = pd.Series(np.concatenate(all_true))
        metrics = evaluate_regression(y_true, y_pred)

        ep_time = time.time() - t_ep
        marker = ""
        if metrics["sigma_nmad"] < best_nmad:
            best_nmad = metrics["sigma_nmad"]
            torch.save(model.state_dict(), out / "best_model.pt")
            marker = " *best*"

        logger.info(
            f"Epoch {ep}/{epochs}  "
            f"loss={train_loss:.5f}  "
            f"nmad={metrics['sigma_nmad']:.5f}  "
            f"bias={metrics['bias']:.5f}  "
            f"outlier%={metrics['f_outlier_percent']:.2f}  "
            f"[{ep_time:.0f}s/ep]{marker}",
        )

    torch.save(model.state_dict(), out / "last_model.pt")

    report = {
        "model_type": "pytorch",
        "best_nmad": best_nmad,
        "val_metrics": metrics,
        "config": cfg.model_dump(exclude={"estimator"}),
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"\nDone! Best nmad: {best_nmad:.5f}")
    logger.info(f"Saved to {out.resolve()}")


def _train_sklearn(
    train_path: str, config_path: str, features_path: str, feature_set: str, outdir: str,
) -> None:
    cfg = load_yaml_as(config_path, SklearnTrainConfig)
    fcfg = load_yaml_as(features_path, FeaturesConfig)
    target = fcfg.target
    feature_list = fcfg.feature_sets[feature_set]

    df = read_dataset(train_path)
    selected, missing = feature_selection(df, feature_list)

    X = df[selected]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.val_size,
        random_state=cfg.random_seed,
        shuffle=cfg.shuffle,
    )

    estimator = load_estimator(cfg.estimator)
    best_params = hyperparameter_optimization(estimator, X_train, y_train, cfg)
    model = train_model(estimator, X_train, y_train, best_params)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.pkl")

    report = {
        "feature_set": feature_set,
        "selected_features": selected,
        "missing_features": missing,
        "best_params": best_params,
        "val_metrics": evaluate(model, X_val, y_val),
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Saved to {out.resolve()}")


def train(
    config_path: str,
    outdir: str | None = None,
    train_path: str | None = None,
    features_path: str | None = None,
    feature_set: str | None = None,
) -> None:
    raw_cfg = load_yaml(config_path)
    if isinstance(raw_cfg, dict) and raw_cfg.get("model_type") == "pytorch":
        _train_pytorch(PytorchTrainConfig.model_validate(raw_cfg), outdir)
    else:
        _train_sklearn(
            cast(str, train_path),
            config_path,
            cast(str, features_path),
            cast(str, feature_set),
            outdir,
        )


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False))
def main(config: str) -> None:
    train(config)


if __name__ == "__main__":
    main()
