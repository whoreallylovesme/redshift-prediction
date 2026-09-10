import json
from pathlib import Path

from lib.dataloaders.dataset import read_dataset
from lib.metrics import evaluate_regression


def calc_metrics(preds: str, outdir: str, model_name: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = read_dataset(preds)
    y_true = df["y_true"]
    y_pred = df["y_pred"].to_numpy()

    metrics = evaluate_regression(y_true, y_pred)

    report = {
        "model_name": model_name,
        "test_metrics": metrics,
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
