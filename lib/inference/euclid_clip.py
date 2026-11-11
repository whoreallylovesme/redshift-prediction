from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.dataloaders.euclid_packet import Packet
from lib.train.euclid_clip_finetune import (
    load_config,
    load_finetuned,
    photoz_metrics,
    pool_arrays,
    predict_sets,
    prepare_split,
    save_predictions,
)
from lib.utils.log import log_event
from lib.utils.torch_utils import configure_torch
from lib.utils.utils import check, pick_device


def run_inference(config_path: Path) -> None:
    cfg = load_config(config_path)
    if cfg.inference is None:
        raise ValueError("config needs an inference section")
    section = cfg.inference
    seeds = section.seeds or [cfg.seed]
    state_dir = section.state_dir or cfg.out_dir
    names = [f"{cfg.tag}_{seed}" for seed in seeds]
    for name in names:
        check((state_dir / f"{name}_state.pt").exists(), f"missing state for {name}")
        out = section.out_dir / f"{name}_pred.npz"
        check(not out.exists(), f"exclusive output: {out}")
    section.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    configure_torch(cfg.num_threads, cfg.memory_fraction, device)
    packet = Packet(cfg.packet)
    split = prepare_split(cfg, packet)
    zval = pool_arrays(split.pool, np.flatnonzero(split.mask), split.overrides)
    for name in names:
        model = load_finetuned(state_dir / f"{name}_state.pt", device)
        preds = predict_sets(model, cfg, packet, split, zval, device)
        save_predictions(section.out_dir / f"{name}_pred.npz", preds)
        log_event(
            phase="inference",
            run=name,
            sets={k: int(len(p.object_id)) for k, p in preds.items()},
            zval=photoz_metrics(split.z, preds["zval"].z),
        )
    packet.close()
