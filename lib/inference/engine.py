from lib.inference.regression import inference as inference
from photoz.methods.legacy import (
    build_predictor as build_predictor,
)
from photoz.methods.legacy import (
    register_predictors,
)
from photoz.methods.legacy import (
    run_inference as run_inference,
)

register_predictors()
