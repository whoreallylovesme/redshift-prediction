from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from lib.config.experiment import External
from lib.report import build_report_data
from lib.report.posterior_metadata import PosteriorMetadata
from lib.workflow.experiment import report_record
from photoz.methods import phosphoros
from photoz.report import PHOTOZ
from tools.euclid_q1 import phosphoros as legacy_cli


class PhosphorosMethodTests(TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        self.out = root / "experiment" / "method"
        self.ids = np.array([2**60 + n for n in (1, 3, 5)], dtype=np.int64)
        self.native_axis = phosphoros.AXIS.copy()
        self.native_axis[20] += 1e-12
        preparation = root / "prepared/sample/preparation"
        preparation.mkdir(parents=True)
        self.catalog = preparation / "computational_inputs.fits"
        columns = [fits.Column(name="OBJECT_ID", format="K", array=self.ids[[2, 0]])]
        for band in phosphoros.NORTH.bands:
            columns.extend(
                [
                    fits.Column(
                        name=f"FLUX_{band.upper()}", format="D", unit="uJy", array=[2.0, 4.0]
                    ),
                    fits.Column(
                        name=f"FLUXERR_{band.upper()}", format="D", unit="uJy", array=[0.5, 0.7]
                    ),
                ]
            )
        for name in ("GAL_EBV", "GAL_EBV_ERR"):
            columns.append(fits.Column(name=name, format="D", unit="mag", array=[0.01, 0.02]))
        table = fits.BinTableHDU.from_columns(columns)
        table.writeto(self.catalog)
        pd.DataFrame(
            {
                "object_id": self.ids,
                "field_code": 0,
                "split": "test",
                "n_valid_bands": [3, 0, 2],
                "input_status": [phosphoros.ELIGIBLE, phosphoros.INELIGIBLE, phosphoros.ELIGIBLE],
            }
        ).to_parquet(preparation / "ledger.parquet", index=False)
        resources = root / "resources"
        (resources / "AuxiliaryData/Filters").mkdir(parents=True)
        (resources / "PhosphorosComputeRedshifts").touch(mode=0o755)
        conf = resources / "redshift.conf"
        conf.write_text(
            "catalog-type=fake\nmodel-grid-file=grid\n"
            "galactic-correction-coefficient-grid-file=dust\n"
        )
        grids = [("ModelGrids", "grid"), ("GalacticCorrectionCoefficientGrids", "dust")]
        for folder, name in grids:
            product = resources / "IntermediateProducts/fake" / folder
            product.mkdir(parents=True)
            (product / name).touch()
        self.config = root / "method.yml"
        self.config.write_text(
            yaml.safe_dump(
                {
                    "executables": str(resources),
                    "output_root": str(root / "prepared"),
                    "sides": {
                        "north": {
                            "workspace": str(resources),
                            "model_grid_config": "unused",
                            "galactic_grid_config": "unused",
                            "redshift_config": str(conf),
                            "grid_export": "unused",
                        }
                    },
                    "sets": {"sample": {"side": "north", "photometry": "unused"}},
                    "variants": {"corrected": {"sets": ["sample"], "multiply": {"VIS": 2.0}}},
                }
            )
        )

    def fake_core(self, argv: list[str], log: Path, cwd: Path, threads: int, timeout: int) -> None:
        args = dict(item[2:].split("=", 1) for item in argv[1:])
        self.assertEqual(cwd, self.out / "execution")
        source = Path(args["input-catalog-file"])
        with fits.open(source) as hdus:
            data = hdus[1].data
            self.assertNotIn("truth", data.names)
            scale = 1.0 if source == self.catalog else 2.0
            np.testing.assert_array_equal(data["FLUX_VIS"], np.array([2.0, 4.0]) * scale)
            ids = data["OBJECT_ID"].copy()
        result = Path(args["phz-output-dir"])
        result.mkdir()
        pdf = np.zeros((2, len(phosphoros.AXIS)))
        pdf[0, 82] = 1
        pdf[1, 10] = np.nan
        table = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="ID", format="K", array=ids),
                fits.Column(name="Z-1D-PDF", format="601D", array=pdf),
                fits.Column(name="Z", format="D", array=[1.2, 1.3]),
                fits.Column(name="1DPDF-Peak-Z", format="D", array=[1.3, 1.4]),
                fits.Column(name="Missing-data-flag", format="L", array=[False, True]),
            ]
        )
        declaration = "Z-BINS:{" + ",".join(map(str, self.native_axis)) + "}"
        for i in range(0, len(declaration), 65):
            table.header.add_comment(declaration[i : i + 65])
        table.writeto(result / "phz_cat.fits")

    def test_base_and_variant_preserve_ledger_and_native_estimator(self) -> None:
        base_out = self.out
        original = self.catalog.read_bytes()
        for variant in (None, "corrected"):
            with self.subTest(variant=variant), patch.object(phosphoros, "call", self.fake_core):
                self.out = base_out / str(variant)
                path = phosphoros.predict(self.config, "sample", self.out, variant)
                self.assertEqual(path, self.out / "predictions.parquet")
                frame = pd.read_parquet(path)
                np.testing.assert_array_equal(frame.object_id, self.ids)
                np.testing.assert_array_equal(frame.y_pred, [np.nan, np.nan, 0.82])
                np.testing.assert_array_equal(frame.y_pred, frame.ppub_median)
                self.assertEqual(list(frame.engine_present), [True, False, True])
                self.assertEqual(list(frame.has_prediction), [False, False, True])
                self.assertNotIn("posterior_mass", frame)
                with self.assertRaisesRegex(ValueError, "did not supply posterior metadata"):
                    PosteriorMetadata.read(path)
                with self.assertRaises(FileExistsError):
                    phosphoros.predict(self.config, "sample", self.out, variant)
        self.assertEqual(self.catalog.read_bytes(), original)

    def test_native_posterior_reaches_the_report_with_and_without_truth(self) -> None:
        manifest = Path(__file__).resolve().parents[1] / "configs/flows/phosphoros.yaml"
        options = yaml.safe_load(manifest.read_text())["methods"]["phosphoros_public"]
        method = External.model_validate(options)
        with patch.object(phosphoros, "call", self.fake_core):
            path = phosphoros.predict(
                self.config,
                "sample",
                self.out,
                include_posterior=method.options["include_posterior"],
            )
        meta = PosteriorMetadata.read(path)
        np.testing.assert_array_equal(meta.posterior_grid, self.native_axis)
        frame = pd.read_parquet(path)
        self.assertEqual(frame[meta.posterior_column].isna().tolist(), [True, True, False])
        self.assertEqual(frame[meta.posterior_column][2][82], 1.0)
        record = report_record("Phosphoros", "test", method, str(path), {}, self.out)
        output = build_report_data([record], PHOTOZ, "Native result", strict=True)
        charts = output["rows"][0]["slices"]["All objects"]["charts"]
        self.assertEqual([e["id"] for e in charts["posterior_examples"]], [str(self.ids[2])])
        reference = self.out / "independent.parquet"
        pd.DataFrame({"object_id": self.ids, "z_ref": [1.1, 1.2, 0.82]}).to_parquet(reference)
        record.update(reference=str(reference), reference_truth_column="z_ref")
        with_truth = build_report_data([record], PHOTOZ, "Independent truth", strict=True)
        calibration = with_truth["rows"][0]["slices"]["All objects"]["charts"]["posterior"]
        self.assertEqual(calibration["count"], 1)
        self.assertEqual(calibration["empirical"], [1.0] * 5)
        with self.assertRaisesRegex(ValueError, "run directory"):
            report_record("Phosphoros", "test", method, str(path), {}, self.out / "unrelated")

    def test_all_invalid_posterior_still_has_a_typed_nullable_vector(self) -> None:
        self.out.mkdir(parents=True)
        target = self.out / "empty-pdfs.parquet"
        spec = PosteriorMetadata(posterior_column="pdf", posterior_grid=[0.0, 1.0])
        phosphoros.write_predictions(pd.DataFrame({"pdf": [None, float("nan")]}), target, spec)
        self.assertEqual(PosteriorMetadata.read(target), spec)
        self.assertTrue(pd.read_parquet(target).pdf.isna().all())

    def test_unknown_set_invalid_variant_and_missing_resource_do_not_call_core(self) -> None:
        with patch.object(phosphoros, "call") as runner:
            with self.assertRaisesRegex(ValueError, "unknown sets"):
                phosphoros.predict(self.config, "missing", self.out)
            with self.assertRaisesRegex(ValueError, "unknown variant"):
                phosphoros.predict(self.config, "sample", self.out, "missing")
            self.catalog.unlink()
            with self.assertRaises(FileNotFoundError):
                phosphoros.predict(self.config, "sample", self.out)
            runner.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_legacy_imports_and_subcommands(self) -> None:
        self.assertIs(legacy_cli.PhosphorosConfig, phosphoros.PhosphorosConfig)
        self.assertEqual(
            set(legacy_cli.cli.commands), {"grids", "workspace", "run", "postprocess", "prepare"}
        )
