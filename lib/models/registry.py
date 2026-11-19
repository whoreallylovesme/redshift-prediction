from collections import ChainMap
from collections.abc import Callable, Mapping
from typing import Any

from lib.models.nn import factories as neural
from lib.models.tabular import factories as tabular

NN_FACTORIES: dict[str, Callable[..., Any]] = {
    "fusion_cnn": neural.fusion_cnn,
    "capsnet": neural.capsnet,
    "convnext_tiny": neural.convnext_tiny,
    "efficientnet_b0": neural.efficientnet_b0,
    "vit_tiny": neural.vit_tiny,
}

TABULAR_FACTORIES: dict[str, Callable[..., Any]] = {
    "catboost": tabular.catboost,
    "imputed_catboost": tabular.imputed_catboost,
    "gam": tabular.gam,
    "llr": tabular.llr,
    "mlp": tabular.mlp,
    "rf": tabular.rf,
    "ridge": tabular.ridge,
}

FACTORIES: Mapping[str, Callable[..., Any]] = ChainMap(TABULAR_FACTORIES, NN_FACTORIES)
