import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from prefect.testing.utilities import prefect_test_harness

from runtime.pipeline import pipeline


def recipe(path: Path, **values: Any) -> str:
    path.write_text(yaml.safe_dump(values))
    return str(path)


def table_flow(root: Path) -> tuple[Path, str]:
    pd.DataFrame(
        {"id": [f"{i:04}" for i in range(20)], "x": range(20), "z": np.arange(20) / 40}
    ).to_csv(root / "data.csv", index=False)
    runs = root / "tables"
    config = recipe(
        root / "tabular.yaml",
        pipeline="tabular",
        runs_root=str(runs),
        prepared_root=str(root / "prepared"),
        dataset={
            "input_csv": str(root / "data.csv"),
            "id_column": "id",
            "data_config": recipe(root / "split.yaml", split={"test_size": 0.2, "shuffle": False}),
            "features": recipe(root / "features.yaml", target="z", feature_sets={"colour": ["x"]}),
        },
        models=[
            {
                "name": "linear",
                "config": recipe(root / "ridge.yaml", estimator="ridge"),
                "feature_set": "colour",
            }
        ],
        summaries={"enabled": True},
    )
    return runs, config


def image_flow(root: Path) -> tuple[Path, str]:
    train, held = root / "train_images", root / "held_images"
    for directory, ids in ((train, range(8)), (held, range(8, 12))):
        directory.mkdir()
        lines = []
        for i in ids:
            stem = f"{i:04}"
            lines.append(f"{stem} 0 0 0 0 0 {i / 30}\n")
            np.savez(directory / f"{stem}.npz", image=np.full((8, 8, 4), i, dtype=np.float32))
        (directory / "labels.cat").write_text("".join(lines))
    (held / "0011.npz").write_bytes(b"partial image")
    model = recipe(
        root / "nn_model.yaml",
        model_type="lightning",
        device="cpu",
        data_dir=str(train),
        catalog=str(train / "labels.cat"),
        out_size=8,
        estimator="from torch import nn\nestimator = nn.Sequential("
        "nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))",
        train={
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.001,
            "weight_decay": 0.0,
            "freeze_epochs": 0,
            "train_ratio": 0.75,
            "num_workers": 0,
        },
    )
    out = root / "images"
    config = recipe(
        root / "nn.yaml",
        pipeline="nn",
        runs_root=str(out),
        models=[{"name": "cnn", "config": model}],
        splits=[{"name": "C", "catalog": str(held / "labels.cat"), "data_dir": str(held)}],
        summaries={"enabled": True},
    )
    return out, config


def capsnet_flow(root: Path) -> tuple[Path, str]:
    h5, held = root / "train.h5", root / "valid.h5"
    for path, count, start in ((h5, 6, 0), (held, 3, 8)):
        with h5py.File(path, "w") as handle:
            handle["images"] = np.random.default_rng(1).normal(size=(count, 5, 16, 16)).astype("f4")
            handle["ObjID"] = np.arange(2**60 + start, 2**60 + start + count, dtype=np.int64)
            handle["specObjID"] = np.arange(count) + 100
            handle["specz_redshift"] = np.linspace(0.05, 0.35, count).astype("f4")
    morph = root / "morph.parquet"
    pd.DataFrame({"gz_morpho_binary": [0, 1, 0, 1, 0, 1]}).to_parquet(morph, index=False)
    out = root / "capsnet" / "hayat"
    model = recipe(
        root / "capsnet_model.yaml",
        model_type="lightning",
        model="capsnet",
        dataset="hayat",
        device="cpu",
        h5_path=str(h5),
        morph_parquet=str(morph),
        out_dir=str(out),
        crop_size=16,
        batch_size=2,
        num_workers=0,
        stage_a_epochs=0,
        stage_b_epochs=1,
        weighted_sampling=False,
        inference={"h5_path": str(held), "out": str(out / "preds.parquet")},
    )
    return out, recipe(
        root / "capsnet_flow.yaml",
        pipeline="capsnet",
        runs_root=str(out.parent),
        train={"config": model},
    )


class LegacyPipelineTests(TestCase):
    def test_real_fits_reach_the_report_through_the_public_yaml_entry(self) -> None:
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        with TemporaryDirectory() as tmp, prefect_test_harness():
            root = Path(tmp)
            for scenario, expected in [(table_flow, 4), (image_flow, 3), (capsnet_flow, 3)]:
                with self.subTest(scenario=scenario.__name__):
                    out, config = scenario(root)
                    pipeline(config)
                    data = json.loads((out / "report-data.json").read_text())
                    row = data["rows"][0]
                    self.assertEqual(row["status"], "completed")
                    self.assertTrue(row["paired_id"])
                    self.assertEqual(row["coverage"]["evaluated"], expected)
                    self.assertTrue((out / "report.html").is_file())
                    self.assertTrue((out / "metrics.csv").is_file())
                    frame = pd.read_parquet(row["predictions"])
                    old = json.loads((Path(row["predictions"]).parent / "report.json").read_text())
                    self.assertEqual(row["metrics"], old["test_metrics"])
                    if scenario in (image_flow, capsnet_flow):
                        self.assertTrue(row["history"])
                    if scenario is image_flow:
                        self.assertEqual(row["coverage"]["requested"], 4)
                        self.assertEqual(frame.object_id.tolist(), ["0008", "0009", "0010", "0011"])
                    elif scenario is capsnet_flow:
                        self.assertEqual(frame.objid.tolist(), [2**60 + n for n in (8, 9, 10)])
                    else:
                        self.assertEqual(frame.id.tolist(), ["0016", "0017", "0018", "0019"])
