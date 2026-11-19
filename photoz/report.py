import numpy as np

from lib.report.task import REGRESSION, Floats, Metric, ReportTask
from photoz.metrics import bias, nmad, normalized_residual, outlier_percent


def guides(truth: Floats) -> dict[str, Floats]:
    margin = 0.15 * (1 + truth)
    return {
        "Диагональ": truth,
        "Порог (+)": truth + margin,
        "Порог (−)": truth - margin,
    }


PHOTOZ = ReportTask(
    name="photoz",
    primary="sigma_nmad",
    metrics={
        **REGRESSION.metrics,
        "sigma_nmad": Metric("NMAD", "min", nmad),
        "bias": Metric("Median bias", "zero", bias),
        "f_outlier_percent": Metric("Outliers", "min", outlier_percent, "%"),
    },
    target_label="Эталонное z",
    prediction_label="Предсказанное z",
    residual_label="Δz / (1 + z)",
    residual=normalized_residual,
    valid_target=lambda t: np.isfinite(t) & (t > -1),
    guides=guides,
)
