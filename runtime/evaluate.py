import json
from pathlib import Path

from lib.report.evaluation import numeric, prepare_frame
from lib.report.task import scores
from photoz.report import PHOTOZ


def calc_metrics(preds: str, outdir: str, model_name: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    frame, _ = prepare_frame({"predictions": preds}, PHOTOZ)
    valid = frame.loc[frame._evaluated]
    report = {
        "model_name": model_name,
        "test_metrics": scores(PHOTOZ, numeric(valid._truth), numeric(valid._pred)),
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
