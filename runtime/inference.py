from pathlib import Path
from typing import cast

import click
import joblib
import pandas as pd

from lib.config_models import FeaturesConfig, PytorchTrainConfig, SklearnTrainConfig
from lib.dataset import read_dataset
from lib.log import get_logger
from lib.utils import load_yaml, load_yaml_as

logger = get_logger()



def _inference_pytorch(
    model_path: str, cfg: PytorchTrainConfig, data_dir: str, catalog: str | None, out: str,
) -> None:
    import torch

    from lib.nn.npz import compute_stats, load_npz, parse_catalog

    scope: dict[str, object] = {}
    exec(cfg.estimator, scope)
    estimator = scope["estimator"]
    if not isinstance(estimator, torch.nn.Module):
        raise TypeError("Loaded object is not a torch.nn.Module")
    model = estimator

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
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
                batch_tensors.append(load_npz(fp, mean, std, crop_size=crop_size))
                loaded_paths.append(fp)
            except Exception as exc:
                skipped_files.append((fp.name, f"{type(exc).__name__}: {exc}"))
        if not batch_tensors:
            continue

        batch = torch.cat(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            preds = model(batch).squeeze(-1).cpu().numpy()
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
    if config_path is not None:
        raw_cfg = load_yaml(config_path)
        if isinstance(raw_cfg, dict) and raw_cfg.get("model_type") == "lightning":
            cfg = PytorchTrainConfig.model_validate(raw_cfg)
            out_dir = cfg.out_dir or f"nn_runs/{Path(config_path).stem}"
            model_path = model_path or f"{out_dir}/best_model.pt"
            out = out or f"{out_dir}/preds.parquet"
            data_dir = data_dir or cfg.data_dir
            catalog = catalog if catalog is not None else cfg.catalog
            _inference_pytorch(
                cast(str, model_path), cfg, cast(str, data_dir), catalog, cast(str, out),
            )
            return
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
        cast(str, out),
    )


@click.command()
@click.option(
    "--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False),
)
def main(config_path: str) -> None:
    inference(config_path=config_path)


if __name__ == "__main__":
    main()
