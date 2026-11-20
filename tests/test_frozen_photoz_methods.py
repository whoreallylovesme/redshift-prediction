import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from photoz.data.weak import load_prepared
from photoz.methods import morph_ensemble as morph


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PreparedCohortTests(TestCase):
    def recipe(self, root: Path, large: bool, blocked: bool = False) -> Path:
        ids = [2**60 + 2, 2**60 + 1]
        frozen = pd.DataFrame(
            {
                "object_id": ids,
                "tile": [2, 1],
                "split": ["valid", "train"],
                "z_train": [0.1, 0.4],
                "label_source": ["phz_quality", "spe_halpha_quality"],
                "sample_weight": [0.3, 1.0],
            }
        )
        sources = {
            "manifest": frozen,
            "observed": frozen.iloc[::-1].assign(flux_vis=[3.0, 8.0]),
            "validation": frozen.iloc[[0]][["object_id", "tile"]],
            "exclusions": pd.DataFrame({"object_id": [2**60], "tile": [3]}),
            "identities": pd.DataFrame({"object_id": ids[:1] if blocked else [2**60 + 4]}),
        }
        opts = {}
        hashes = {}
        for key, frame in sources.items():
            path = root / f"{key}.parquet"
            frame.to_parquet(path, index=False)
            opts[key] = path.name
            hashes[key] = digest(path)
        protocol = root / "protocol.json"
        outputs = {
            "valid_memberships": {"sha256": hashes["validation"]},
            "test_exclusions": {"sha256": hashes["exclusions"]},
        }
        if large:
            outputs["fitting_excluded_ids"] = {"sha256": hashes["identities"]}
        protocol.write_text(json.dumps({"outputs": outputs}))
        receipt = root / "provenance.json"
        obj = {
            "output": {"sha256": hashes["observed"]},
            "manifest": {"sha256": hashes["manifest"]},
            "desi_used_as_training_target": False,
            "phz_unif_used": False,
            "output_flux_unit": "uJy",
            "sources_and_protocol": {"protocol": {"sha256": digest(protocol)}},
            "exclusions_sha256": hashes["exclusions"],
        }
        identities = opts.pop("identities")
        if large:
            obj["identity_exclusions_sha256"] = hashes["identities"]
            opts["identity_exclusions"] = identities
        receipt.write_text(json.dumps(obj))
        options = dict(
            opts,
            cohort="large" if large else "weak",
            provenance=receipt.name,
            provenance_sha256=digest(receipt),
            protocol=protocol.name,
            protocol_sha256=digest(protocol),
        )
        config = root / "recipe.yml"
        config.write_text(yaml.safe_dump(options))
        return config

    def test_original_membership_owner_weights_and_unordered_ids(self) -> None:
        for large in (True, False):
            with self.subTest(large=large), TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg = self.recipe(root, large)
                out = load_prepared(cfg)
                self.assertEqual(out.object_id.tolist(), [2**60 + 1, 2**60 + 2])
                self.assertEqual(out.sample_weight.tolist(), [1, 0.3])
                self.assertEqual(out.split.tolist(), ["train", "valid"])
                (root / "exclusions.parquet").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    load_prepared(cfg)

    def test_large_requires_complete_independent_identity_embargo(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.recipe(root, True, blocked=True)
            with self.assertRaisesRegex(ValueError, "overlapping fitting identity"):
                load_prepared(cfg)
            spec = yaml.safe_load(cfg.read_text())
            del spec["identity_exclusions"]
            cfg.write_text(yaml.safe_dump(spec))
            with self.assertRaisesRegex(ValueError, "bound fitting identity"):
                load_prepared(cfg)


class MorphMethodTests(TestCase):
    def test_normalization_and_ids(self) -> None:
        x = np.full((1, 4, 64, 64), 3.0, dtype=np.float32)
        scales = np.array([1, 2, 3, 4], dtype=np.float32)
        np.testing.assert_allclose(
            morph.normalize_images(x, scales)[0, :, 0, 0], np.arcsinh(3.0 / scales) / 3
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            morph.normalize_images(x, np.array([0, 1, 2, 3]))
        with self.assertRaisesRegex(ValueError, "64"):
            morph.normalize_images(x[:, :2], scales)

    def test_lightning_prediction_aligns_exact_shard_ids(self) -> None:
        class PixelModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None, None]:
                return x[:, 0, 0, 0], None, None

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "prepared.npz"
            ids = np.array([2**60 + 7, 2**60 + 1], dtype=np.int64)
            x = np.stack([np.full((4, 64, 64), val, np.float32) for val in (0, 4)])
            np.savez(shard, ids=ids, X=x)
            resource = morph.PinnedFile(path=Path(shard.name), sha256=digest(shard))
            recipe = morph.MorphRecipe(checkpoint_root=root, reference_dir=root, sets={})
            values = morph.predict_shards(
                root,
                [resource],
                "ids",
                "X",
                pd.Index(ids[::-1]),
                morph.MorphPredictor(cast(morph.CapsNet, PixelModel())),
                np.ones(4, dtype=np.float32),
                recipe,
            )
            logits = np.arcsinh(np.array([4, 0], np.float32)) / 3
            np.testing.assert_allclose(values, 3 * np.exp(logits) / (np.exp(logits) + 1))

    def test_reference_verification_precedes_all_target_predictions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "ids.parquet"
            ids = [2**60 + 3, 2**60 + 1]
            pd.DataFrame({"object_id": ids}).to_parquet(manifest, index=False)
            bound = {"path": manifest.name, "sha256": digest(manifest)}
            cfg = root / "recipe.yml"
            cfg.write_text(
                yaml.safe_dump(
                    {
                        "checkpoint_root": ".",
                        "reference_dir": "refs",
                        "sets": {"held": {"manifest": bound, "shards": [bound]}},
                    }
                )
            )
            calls: list[bool] = []

            def fake(
                shard_root: Path,
                shards: list[morph.PinnedFile],
                ids_key: str,
                image_key: str,
                requested: pd.Index,
                model: Any,
                scales: np.ndarray,
                recipe: morph.MorphRecipe,
            ) -> np.ndarray:
                seed = model.network.seed
                reference = tuple(requested) == morph.REF_IDS
                calls.append(reference)
                if reference:
                    return np.array(morph.REF_VALUES[seed])
                return np.full(len(requested), seed / 100)

            def network(path: Path, seed: int) -> tuple[Any, np.ndarray]:
                net: Any = torch.nn.Identity()
                net.seed = seed
                return net, np.ones(4, dtype=np.float32)

            with (
                patch.object(morph, "load_network", side_effect=network),
                patch.object(morph, "predict_shards", side_effect=fake),
            ):
                frame = pd.read_parquet(morph.predict(cfg, "held", root / "out"))
            self.assertEqual(calls, [True] * 4 + [False] * 4)
            self.assertEqual(frame.object_id.tolist(), ids)
            np.testing.assert_allclose(frame.y_pred, [2.025, 2.025])
