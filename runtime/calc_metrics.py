import json
from pathlib import Path

import click

from lib.dataset import read_dataset
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


@click.command()
@click.option("--preds", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--outdir", required=True, type=click.Path(file_okay=False))
@click.option("--model-name", "model_name", required=True)
def main(preds: str, outdir: str, model_name: str) -> None:
    calc_metrics(preds, outdir, model_name)


if __name__ == "__main__":
    main()
