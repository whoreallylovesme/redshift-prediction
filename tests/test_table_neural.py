from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd
import torch
import yaml

from photoz.data.euclid_transforms import BANDS
from photoz.models.tabular import Architecture, TableRegressor
from photoz.training.tabular import read_projection, run_method


def fixture(path: Path) -> None:
    rng = np.random.default_rng(73)
    n = 8
    np.savez(
        path,
        object_id=np.arange(n, dtype=np.int64) + 2**57,
        field_code=np.ones(n, np.uint8),
        role_bitmask=np.array([2] * 5 + [4] * 3, dtype=np.uint8),
        z=np.linspace(0.04, 0.9, n),
        fit_indices=np.arange(5, dtype=np.int64),
        valid_indices=np.array([6, 5, 7], dtype=np.int64),
        raw_flux_ujy=rng.normal(size=(n, 14)),
        raw_error_ujy=rng.uniform(0.1, 0.9, size=(n, 14)),
        table_mask=rng.uniform(size=(n, 14)) < 0.8,
        table_descriptor=rng.normal(size=(14, 18)).astype("f4"),
        bands=np.array(BANDS),
    )


class NeuralTableTests(TestCase):
    def test_independent_normalization_uses_only_fit_measurements(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.npz"
            fixture(path)
            original = read_projection(path)
            with np.load(path, allow_pickle=False) as source:
                data = dict(source)
            data["raw_flux_ujy"][original.valid] += 1e6
            np.savez(path, **data)
            changed = read_projection(path)
            np.testing.assert_array_equal(original.mean, changed.mean)
            torch.testing.assert_close(original.values[original.fit], changed.values[changed.fit])
            data["valid_indices"] = np.array([1, 5, 7], np.int64)
            np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, "partition"):
                read_projection(path)

    def test_both_recipes_preserve_head_initialization_ids_and_selected_state(self) -> None:
        seed = 52
        arms: tuple[Architecture, Architecture] = ("resmlp", "table_transformer")
        first, second = (TableRegressor(arm, seed) for arm in arms)
        for a, b in zip(
            first.photoz_head.parameters(), second.photoz_head.parameters(), strict=True
        ):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "input.npz"
            fixture(path)
            data = read_projection(path)
            for arm in arms:
                recipe = root / f"{arm}.yaml"
                recipe.write_text(
                    yaml.safe_dump(
                        dict(
                            architecture=arm,
                            projection=str(path),
                            epochs=2,
                            batch_size=3,
                            seed=seed,
                        )
                    )
                )
                frame = pd.read_parquet(run_method(recipe, root / arm))
                np.testing.assert_array_equal(frame.object_id, data.ids[data.valid])
                np.testing.assert_array_equal(frame.y_true, data.truth[data.valid])
                model = TableRegressor(arm, seed)
                model.load_state_dict(
                    torch.load(root / arm / "selected_state.pt", weights_only=True)
                )
                model.eval()
                with torch.inference_mode():
                    values = model(data.values[data.valid], data.mask[data.valid], data.descriptor)
                np.testing.assert_allclose(
                    frame.y_pred, np.expm1(values.double().numpy()), atol=1e-6, rtol=1e-6
                )
