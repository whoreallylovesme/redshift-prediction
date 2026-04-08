from pathlib import Path
from typing import Any

import click
import joblib
import numpy as np
import pandas as pd
import yaml

from lib.dataset import read_dataset


def load_yaml(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _inference_pytorch(
    model_path: str, cfg: dict[str, Any], data_dir: str, catalog: str | None, out: str,
) -> None:
    import torch

    from lib.nn.dataloader_photoz import compute_stats, parse_catalog
    from lib.nn.nn_inference_photoz import load_npz

    # Load model from estimator code
    scope: dict[str, Any] = {}
    exec(cfg["estimator"], scope)
    model: torch.nn.Module = scope["estimator"]

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Device: {device}")

    # Unfreeze for inference (no-op functionally, but needed for correct model structure)
    for p in model.parameters():
        p.requires_grad = False

    paths = sorted(Path(data_dir).glob("*.npz"))
    if not paths:
        raise ValueError(f"No .npz files in {data_dir}")

    norm_dir = Path(cfg.get("norm_from", data_dir))
    norm_paths = sorted(norm_dir.glob("*.npz"))
    print(f"Computing normalization stats from {norm_dir} ({len(norm_paths)} files)...")
    mean, std = compute_stats(norm_paths)

    z_true_map: dict[str, float] = {}
    if catalog is not None:
        z_true_map = parse_catalog(catalog)

    from tqdm import tqdm

    batch_size = int(cfg.get("train", {}).get("batch_size", 32))
    results: list[dict[str, object]] = []
    n_batches = (len(paths) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(paths), batch_size), total=n_batches, desc="Inference"):
        batch_paths = paths[i : i + batch_size]
        batch = torch.cat(
            [load_npz(fp, mean, std) for fp in batch_paths], dim=0
        ).to(device)
        with torch.no_grad():
            preds = model(batch).squeeze(-1).cpu().numpy()
        if preds.ndim == 0:
            preds = preds.reshape(1)

        for fp, pred in zip(batch_paths, preds, strict=True):
            row: dict[str, object] = {"y_pred": float(pred)}
            if fp.stem in z_true_map:
                row["y_true"] = z_true_map[fp.stem]
            results.append(row)

    df = pd.DataFrame(results)
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} predictions to {out}")

    if "y_true" in df.columns:
        from lib.metrics import evaluate_regression
        metrics = evaluate_regression(df["y_true"], df["y_pred"].to_numpy(dtype=np.float64))
        for k, v in metrics.items():
            print(f"  {k}: {v:.5f}")


def _inference_sklearn(
    model_path: str, test_path: str, features_path: str, feature_set: str, out: str,
) -> None:
    fcfg = load_yaml(features_path)
    target = fcfg["target"]
    feature_list = fcfg["feature_sets"][feature_set]

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


def inference(
    model_path: str,
    out: str,
    config_path: str | None = None,
    test_path: str | None = None,
    features_path: str | None = None,
    feature_set: str | None = None,
    data_dir: str | None = None,
    catalog: str | None = None,
) -> None:
    if config_path is not None:
        cfg = load_yaml(config_path)
        if cfg.get("model_type") == "pytorch":
            assert data_dir is not None, "data_dir is required for pytorch models"
            _inference_pytorch(model_path, cfg, data_dir, catalog, out)
            return

    assert test_path is not None, "test_path is required for sklearn models"
    assert features_path is not None, "features_path is required for sklearn models"
    assert feature_set is not None, "feature_set is required for sklearn models"
    _inference_sklearn(model_path, test_path, features_path, feature_set, out)


@click.command()
@click.option("--model", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--test", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--features", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--feature-set", "feature_set", default=None)
@click.option("--config", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--data-dir", default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--catalog", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", required=True, type=click.Path(dir_okay=False))
def main(
    model: str, test: str | None, features: str | None, feature_set: str | None,
    config: str | None, data_dir: str | None, catalog: str | None, out: str,
) -> None:
    inference(model, out, config, test, features, feature_set, data_dir, catalog)


if __name__ == "__main__":
    main()
