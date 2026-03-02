from pathlib import Path

import click

from lib.dataset import read_dataset
from lib.metrics import evaluate_regression
from lib.plots import redshift_scatter


@click.command()
@click.option("--preds", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--outdir", required=True, type=click.Path(file_okay=False))
@click.option("--model-name", "model_name", required=True)
def main(preds: str, outdir: str, model_name: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    df = read_dataset(preds)

    y_true = df["y_true"]
    y_pred = df["y_pred"].to_numpy()

    metrics = evaluate_regression(y_true, y_pred)

    redshift_scatter(y_true, y_pred, plots / "scatter.png", model_name, 0, 3)

    (out / "report.html").write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Inference report</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
img {{ max-width: 900px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 24px; }}
code {{ background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }}
</style>
</head>
<body>
<h1>Inference report</h1>

<p><b>RMSE:</b> <code>{metrics["rmse"]:.6g}</code></p>
<p><b>MAE:</b> <code>{metrics["mae"]:.6g}</code></p>
<p><b>R2:</b> <code>{metrics["r2"]:.6g}</code></p>
<p><b>f_outlier_percent:</b> <code>{metrics["f_outlier_percent"]:.6g}</code></p>
<p><b>bias:</b> <code>{metrics["bias"]:.6g}</code></p>
<p><b>sigma_nmad:</b> <code>{metrics["sigma_nmad"]:.6g}</code></p>

<h2>Scatter</h2>
<img src="plots/scatter.png">

</body>
</html>
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
