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
from tools.euclid_q1.phosphoros import cli


class PhosphorosMethodTests(TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        self.out = root / "experiment" / "method"
        self.ids = np.array([2**60 + n for n in (1, 3, 5)], dtype=np.int64)
        self.native_axis = phosphoros.AXIS.copy()
        self.native_axis[82] += 1e-12
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
                            "redshift_config": str(conf),
                        }
                    },
                    "sets": {"sample": {"side": "north"}},
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
                np.testing.assert_array_equal(frame.y_pred, [np.nan, np.nan, self.native_axis[82]])
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
        from lib.config import Experiment, load_flow

        experiment = load_flow(Path("examples/configs/euclid.yaml"), check=True)
        assert isinstance(experiment, Experiment)
        method = experiment.methods["phosphoros"]
        assert isinstance(method, External)
        method = method.model_copy(
            update={"report": method.report.model_copy(update={"id_column": "object_id"})}
        )
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
        record = report_record("Phosphoros", "test", method, str(path), None, self.out)
        output = build_report_data([record], PHOTOZ, "Native result", strict=True)
        charts = output["rows"][0]["slices"]["All objects"]["charts"]
        self.assertEqual([e["id"] for e in charts["posterior_examples"]], [str(self.ids[2])])
        reference = self.out / "independent.parquet"
        pd.DataFrame({"object_id": self.ids, "z_ref": [1.1, 1.2, self.native_axis[82]]}).to_parquet(
            reference
        )
        record.update(reference=str(reference), reference_truth_column="z_ref")
        with_truth = build_report_data([record], PHOTOZ, "Independent truth", strict=True)
        calibration = with_truth["rows"][0]["slices"]["All objects"]["charts"]["posterior"]
        self.assertEqual(calibration["count"], 1)
        self.assertEqual(calibration["empirical"], [1.0] * 5)
        with self.assertRaisesRegex(ValueError, "run directory"):
            report_record("Phosphoros", "test", method, str(path), None, self.out / "unrelated")

    def test_all_invalid_posterior_still_has_a_typed_nullable_vector(self) -> None:
        self.out.mkdir(parents=True)
        target = self.out / "empty-pdfs.parquet"
        spec = PosteriorMetadata(posterior_column="pdf", posterior_grid=[0.0, 1.0])
        phosphoros.write_predictions(pd.DataFrame({"pdf": [None, float("nan")]}), target, spec)
        self.assertEqual(PosteriorMetadata.read(target), spec)
        self.assertTrue(pd.read_parquet(target).pdf.isna().all())

    def test_subset_rejects_lossy_ids_and_preserves_decimal_int64_strings(self) -> None:
        path = self.config.with_suffix(".parquet")
        cfg = phosphoros.SetConfig(side="north", photometry=self.catalog, ids_table=path)
        raw = {"object_id": self.ids}
        pd.DataFrame({"object_id": self.ids[1::-1].astype(str)}).to_parquet(path)
        np.testing.assert_array_equal(phosphoros.select_rows(raw, cfg), [1, 0])
        pd.DataFrame({"object_id": self.ids[:1].astype(float)}).to_parquet(path)
        with self.assertRaisesRegex(ValueError, "exact"):
            phosphoros.select_rows(raw, cfg)

    def test_invalid_prepared_ledger_is_rejected_before_native_execution(self) -> None:
        ledger = self.catalog.parent / "ledger.parquet"
        original = pd.read_parquet(ledger)
        overflow = self.ids.astype(np.uint64)
        overflow[1] = np.uint64(2**63)
        for changed, error in (
            (original.assign(object_id=overflow), "exact"),
            (original.assign(field_code=1), "outside north"),
            (original.assign(input_status="stale"), "input status"),
        ):
            with self.subTest(error=error), patch.object(phosphoros, "call") as core:
                changed.to_parquet(ledger, index=False)
                with self.assertRaisesRegex(ValueError, error):
                    phosphoros.predict(self.config, "sample", self.out)
                core.assert_not_called()
                self.assertFalse(self.out.exists())

    def test_grids_refuse_existing_galactic_grid_and_export_before_native_execution(self) -> None:
        cfg = phosphoros.load_config(self.config)
        side = cfg.sides["north"]
        side.model_grid_config = self.config.with_name("model.conf")
        side.model_grid_config.write_text("catalog-type=fake\noutput-model-grid=new-grid\n")
        side.galactic_grid_config = self.config.with_name("gal.conf")
        side.galactic_grid_config.write_text(
            "catalog-type=fake\nmodel-grid-file=new-grid\noutput-galactic-correction-coefficient-grid=dust\n"
        )
        side.grid_export = self.config.with_suffix(".fits")
        correction = (
            side.workspace / "IntermediateProducts/fake/GalacticCorrectionCoefficientGrids/dust"
        )
        with patch.object(phosphoros, "call") as core:
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                phosphoros.build_grids(cfg, "north")
            correction.unlink()
            side.grid_export.write_bytes(b"existing export")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                phosphoros.build_grids(cfg, "north")
            self.assertEqual(side.grid_export.read_bytes(), b"existing export")
            core.assert_not_called()

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

    def test_preparation_subcommands(self) -> None:
        self.assertEqual(
            set(cli.commands), {"grids", "workspace", "run", "postprocess", "prepare"}
        )
