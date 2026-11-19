import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd
import yaml

from photoz.data.weak_targets import assign_weak_targets, prepare_targets


class WeakTargetTests(TestCase):
    def setUp(self) -> None:
        self.base = 2**60
        self.owner = self.base + 99
        ids = [self.base + i for i in range(1, 9)]
        self.population = pd.DataFrame(
            {
                "object_id": ids,
                "tile": [1, 2, 3, 4, 5, self.owner, 7, 8],
                "phz_median": [0.1, 1.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            }
        )
        self.halpha = pd.DataFrame(
            {
                "object_id": [ids[1], ids[2], ids[7], self.base + 9],
                "tile": [2, 3, 8, 9],
                "spe_z": [0.2, 2.3, 2.8, 0.9],
            }
        ).astype({"object_id": "uint64"})
        self.exclusions = pd.DataFrame({"object_id": [ids[3]], "tile": [5]})
        self.valid = pd.DataFrame({"object_id": [ids[6]], "tile": [self.owner]})
        self.ambiguous = pd.DataFrame({"object_id": [ids[2]]})
        self.embargo = pd.DataFrame({"object_id": [ids[7]]})

    def test_spatial_splits_teacher_priority_ambiguity_and_identity_embargo(self) -> None:
        args = self.population, self.halpha, self.exclusions, self.valid, self.ambiguous
        weak = assign_weak_targets(*args, cohort="weak")
        large = assign_weak_targets(*args, cohort="large", identity_exclusions=self.embargo)
        self.assertEqual(list(weak.object_id), [self.base + i for i in (1, 2, 3, 7, 8, 9)])
        self.assertEqual(list(large.object_id), [self.base + i for i in (1, 2, 3, 7, 9)])
        self.assertEqual(list(large.z_train), [0.1, 0.2, 0.3, 0.7, 0.9])
        self.assertEqual(list(large.sample_weight), [0.3, 1.0, 0.3, 0.3, 1.0])
        self.assertEqual(list(large.split), ["train", "train", "train", "valid", "train"])
        self.assertEqual(int(large.tile[3]), self.owner)
        self.assertEqual(self.population.tile[6], 7)
        with self.assertRaisesRegex(ValueError, "identity embargo"):
            assign_weak_targets(*args, cohort="large")
        with self.assertRaisesRegex(ValueError, "exact nonnegative"):
            assign_weak_targets(
                self.population.astype({"object_id": float}), *args[1:], cohort="weak"
            )

    def test_fresh_manifest_from_only_pinned_candidates_and_target_free_protocol(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frames = {
                "population": self.population,
                "halpha": self.halpha,
                "ambiguous_spe": self.ambiguous,
                "test_exclusions": self.exclusions,
                "valid_memberships": self.valid,
                "fitting_excluded_ids": self.embargo,
                "all_desi_identity_exclusions": self.embargo,
            }
            bound = {}
            for name, table in frames.items():
                path = root / f"{name}.parquet"
                table.to_parquet(path, index=False)
                bound[name] = {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            sources = root / "inputs.yaml"
            sources.write_text(
                yaml.safe_dump(
                    {name: bound[name] for name in ["population", "halpha", "ambiguous_spe"]}
                )
            )
            protocol = root / "protocol.json"
            protocol.write_text(
                json.dumps({"outputs": bound, "desi_used_for_training_or_selection": False})
            )
            config = root / "large.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "cohort": "large",
                        "protocol": protocol.name,
                        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                        "validation": "valid_memberships.parquet",
                        "exclusions": "test_exclusions.parquet",
                        "identity_exclusions": "fitting_excluded_ids.parquet",
                    }
                )
            )
            out = root / "output/manifest.parquet"
            self.assertEqual(prepare_targets(config, sources, out), out)
            self.assertEqual(
                list(pd.read_parquet(out).object_id), [self.base + i for i in (1, 2, 3, 7, 9)]
            )
            with self.assertRaises(FileExistsError):
                prepare_targets(config, sources, out)
            (root / "ambiguous_spe.parquet").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                prepare_targets(config, sources, out.with_name("new.parquet"))
