import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase

import numpy as np
import pandas as pd
import torch
import yaml

from photoz.models.density import posterior_quantiles
from photoz.models.masked_photometry import MaskedFluxEncoder
from photoz.models.prototype_graph import PrototypeGraphDensity
from photoz.training.masked_photometry import run_method as masked_method
from photoz.training.prototype_graph import run_method as graph_method


def masked_projection() -> dict[str, Any]:
    rng = np.random.default_rng(34)
    mask = np.ones((9, 4), np.bool_)
    mask[2, 1:] = False
    mask[-1, :1] = False
    flux, error = rng.normal(size=mask.shape), rng.uniform(0.05, 0.2, mask.shape)
    flux[~mask], error[~mask] = np.nan, np.nan
    return dict(
        object_id=2**56 + np.arange(9),
        bands=np.array(["r", "i", "z", "Y"]),
        fit_indices=np.arange(6),
        valid_indices=np.array([8, 6, 7]),
        raw_flux_ujy=flux,
        raw_error_ujy=error,
        table_mask=mask,
    )


def graph_projection() -> dict[str, Any]:
    rng = np.random.default_rng(60)
    bank = np.arange(17)
    return dict(
        object_id=2**56 + np.arange(21),
        fit_indices=np.arange(18),
        valid_indices=np.array([20, 18, 19]),
        bank_rows=bank,
        embedding=rng.normal(size=(21, 64)).astype("f4"),
        z_edges=np.linspace(0, 3, 401),
        z=np.linspace(0, 3, 21),
        bank_neighbors=np.stack([bank[bank != i] for i in bank]),
        bank_weights=rng.uniform(0.1, 2, (17, 16)).astype("f4"),
        query_neighbors=np.stack([rng.choice(bank, 4, replace=False) for _ in range(21)]),
        query_weights=rng.uniform(0.1, 2, (21, 4)).astype("f4"),
    )


class NativeLikelihoodTests(TestCase):
    def check_selection(self, out: Path) -> dict:
        report: dict[str, Any] = json.loads((out / "train_report.json").read_text())
        losses = [row["val/nll"] for row in report["history"]]
        self.assertTrue(np.isfinite(losses).all())
        self.assertEqual(len(losses), 2)
        self.assertEqual(report["selected_epoch"], np.argmin(losses) + 1)
        self.assertTrue((out / "last_state.pt").is_file())
        return report

    def test_ssl_selected_raw_flux_reconstruction_has_no_hidden_input(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = masked_projection()
            path, recipe, out = root / "ssl.npz", root / "ssl.yaml", root / "fit"
            np.savez(path, **data)
            recipe.write_text(
                yaml.safe_dump(
                    dict(
                        projection=str(path),
                        batch_size=2,
                        epochs=2,
                        mask_fraction=0.5,
                        hidden_dim=8,
                        heads=2,
                        layers=1,
                    )
                )
            )
            frame = pd.read_parquet(masked_method(recipe, out))
            report = self.check_selection(out)
            self.assertEqual(report["ineligible_fit"], 1)
            train = data["fit_indices"]
            scale = np.maximum(
                np.nanquantile(abs(data["raw_flux_ujy"][train]), 0.9, axis=0),
                np.nanmedian(data["raw_error_ujy"][train], axis=0),
            )
            np.testing.assert_allclose(report["scale_ujy"], scale)
            self.assertTrue(frame.measurement_id.is_unique)
            bands = tuple(data["bands"])
            model = MaskedFluxEncoder(bands, hidden_dim=8, num_heads=2, num_layers=1, dropout=0.1)
            model.load_state_dict(torch.load(out / "selected_state.pt", weights_only=True))
            model.eval()
            rows = data["valid_indices"]
            valid = torch.from_numpy(data["table_mask"][rows])
            flux = torch.tensor(data["raw_flux_ujy"][rows] / scale, dtype=torch.float32)
            err = torch.tensor(data["raw_error_ujy"][rows] / scale, dtype=torch.float32)
            hidden = torch.zeros_like(valid)
            lookup = dict(zip(data["object_id"][rows], range(len(rows)), strict=True))
            for record in frame.itertuples():
                hidden[lookup[record.object_id], bands.index(record.band)] = True
            with torch.inference_mode():
                pred = model(flux, err, valid, masked=hidden, bands=bands)
                hidden_perturbed = model(
                    flux.masked_fill(hidden, 1e4),
                    err.masked_fill(hidden, 10),
                    valid,
                    masked=hidden,
                    bands=bands,
                )
            torch.testing.assert_close(pred.mean, hidden_perturbed.mean, rtol=0, atol=0)
            q, b = np.nonzero(hidden.numpy())
            np.testing.assert_array_equal(frame.object_id, data["object_id"][rows[q]])
            np.testing.assert_allclose(frame.y_pred, pred.mean[q, b].numpy() * scale[b], atol=1e-6)
            np.testing.assert_allclose(frame.y_true, data["raw_flux_ujy"][rows[q], b])
            self.assertTrue((frame.observed_sigma > 0).all())

    def test_graph_reloads_density_head_with_training_only_bank(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = graph_projection()
            path, recipe, out = root / "graph.npz", root / "graph.yaml", root / "fit"
            np.savez(path, **data)
            recipe.write_text(yaml.safe_dump(dict(projection=str(path), batch_size=2, epochs=2)))
            frame = pd.read_parquet(graph_method(recipe, out))
            self.check_selection(out)
            rows, bank = data["valid_indices"], data["bank_rows"]
            model = PrototypeGraphDensity(
                torch.tensor(data["embedding"][bank]),
                torch.tensor(data["bank_neighbors"]),
                torch.tensor(data["bank_weights"]),
                data["z_edges"],
                context_mode="graph",
            )
            model.load_state_dict(torch.load(out / "selected_state.pt", weights_only=True))
            model.eval()
            with torch.inference_mode():
                pred = model(
                    torch.tensor(data["embedding"][rows]),
                    torch.tensor(data["query_neighbors"][rows]),
                    torch.tensor(data["query_weights"][rows]),
                )
            np.testing.assert_array_equal(frame.object_id, data["object_id"][rows])
            np.testing.assert_allclose(
                frame[["q16", "y_pred", "q84"]],
                posterior_quantiles(pred.bin_mass.numpy(), data["z_edges"]),
                atol=1e-6,
            )
            data["bank_rows"][0] = rows[0]
            np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, "training-row"):
                graph_method(recipe, root / "invalid")
