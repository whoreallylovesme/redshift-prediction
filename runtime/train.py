import json
import time
from pathlib import Path
from typing import Any

import click
import joblib
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from lib.dataset import read_dataset
from lib.estimator_loader import load_estimator
from lib.training import (
    evaluate,
    feature_selection,
    hyperparameter_optimization,
    train_model,
)


def load_yaml(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _train_pytorch(cfg: dict[str, Any], outdir: str) -> None:
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
    scope: dict[str, Any] = {}
    exec(cfg["estimator"], scope)
    model: nn.Module = scope["estimator"]

    tcfg = cfg["train"]
    epochs = int(tcfg["epochs"])
    batch_size = int(tcfg["batch_size"])
    lr = float(tcfg["lr"])
    weight_decay = float(tcfg["weight_decay"])
    freeze_epochs = int(tcfg["freeze_epochs"])
    seed = int(tcfg.get("seed", 42))
    num_workers = int(tcfg.get("num_workers", 4))
    weighted_sampling = bool(tcfg.get("weighted_sampling", False))
    z_weight_alpha = float(tcfg.get("z_weight_alpha", 0.0))
    label_noise = float(tcfg.get("label_noise", 0.0))
    head_prefix = str(tcfg.get("head_prefix", "classifier"))

    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    max_samples = tcfg.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)

    train_loader, val_loader = get_photoz_dataloaders(
        cfg["data_dir"], cfg["catalog"],
        batch_size=batch_size, num_workers=num_workers,
        train_ratio=float(tcfg.get("train_ratio", 0.8)), seed=seed,
        weighted_sampling=weighted_sampling,
        max_samples=max_samples,
    )

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

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
            print("\n>>> Backbone unfrozen!")
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

        print(
            f"Epoch {ep}/{epochs}  "
            f"loss={train_loss:.5f}  "
            f"nmad={metrics['sigma_nmad']:.5f}  "
            f"bias={metrics['bias']:.5f}  "
            f"outlier%={metrics['f_outlier_percent']:.2f}  "
            f"[{ep_time:.0f}s/ep]{marker}"
        )

    torch.save(model.state_dict(), out / "last_model.pt")

    report = {
        "model_type": "pytorch",
        "best_nmad": best_nmad,
        "val_metrics": metrics,
        "config": {k: v for k, v in cfg.items() if k != "estimator"},
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDone! Best nmad: {best_nmad:.5f}")
    print(f"Saved to {out.resolve()}")


def _train_sklearn(
    train_path: str, config_path: str, features_path: str, feature_set: str, outdir: str,
) -> None:
    cfg = load_yaml(config_path)
    fcfg = load_yaml(features_path)
    target = fcfg["target"]
    feature_list = fcfg["feature_sets"][feature_set]

    split = cfg["split"]
    val_size = float(split["val_size"])
    shuffle = bool(split["shuffle"])
    random_seed = int(split["random_seed"])

    df = read_dataset(train_path)
    selected, missing = feature_selection(df, feature_list)

    X = df[selected]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_seed, shuffle=shuffle,
    )

    estimator = load_estimator(cfg)
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
    print(f"Saved to {out.resolve()}")


def train(
    config_path: str,
    outdir: str,
    train_path: str | None = None,
    features_path: str | None = None,
    feature_set: str | None = None,
) -> None:
    cfg = load_yaml(config_path)
    if cfg.get("model_type") == "pytorch":
        _train_pytorch(cfg, outdir)
    else:
        assert train_path is not None, "--train is required for sklearn models"
        assert features_path is not None, "--features is required for sklearn models"
        assert feature_set is not None, "--feature-set is required for sklearn models"
        _train_sklearn(train_path, config_path, features_path, feature_set, outdir)


@click.command()
@click.option("--train", "train_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--features", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--feature-set", default=None)
@click.option("--outdir", required=True, type=click.Path(file_okay=False))
def main(
    train_path: str | None, config: str, features: str | None,
    feature_set: str | None, outdir: str,
) -> None:
    train(config, outdir, train_path, features, feature_set)


if __name__ == "__main__":
    main()
