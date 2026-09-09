from __future__ import annotations

import logging
import sys
from typing import cast


def get_logger(name: str = "pipeline") -> logging.Logger:
    try:
        from prefect import get_run_logger

        return cast(logging.Logger, get_run_logger())
    except Exception:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
