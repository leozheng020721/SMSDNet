"""Offline persistence and stage-isolation checks for PAN-based NIR recovery."""

import contextlib
from concurrent.futures import Future
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

from align_seo_nir import discover_pairs, resize_coordinate_matrix
import recover_seo_nir_pan as recovery


class RecoveryProductTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "dataset"
        self.output = self.root / "nir_aligned_v1"
        self.source_size = (64, 64)
        self.target_size = (256, 256)
        self.h = resize_coordinate_matrix(self.source_size, self.target_size)
        rng = np.random.default_rng(71)
        visible = rng.integers(300, 2000, (3, 64, 64), dtype=np.uint16)
        self.nir = (4097 + np.indices((64, 64)).sum(axis=0)).astype(np.uint16)
        self.nir[30, 30] = 0
        self.valid = self.nir != 0
        raw = np.concatenate((visible, self.nir[None]))
        self.paths = {
            "msi": self.root / "msi" / "JAX_100" / "JAX_100_13_msi.tif",
            "pan": self.root / "crops" / "JAX_100" / "JAX_100_13_pan.tif",
            "rgb": self.root / "pansharpened_he" / "JAX_100" / "JAX_100_13_rgb.png",
        }
        for path in self.paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            for name, data in (("msi", raw), ("pan", np.full((1, 256, 256), 1000, np.uint16))):
                with rasterio.open(self.paths[name], "w", driver="GTiff", width=data.shape[2],
                                   height=data.shape[1], count=len(data), dtype="uint16", nodata=0,
                                   transform=Affine.translation(500, 900)) as dataset:
                    dataset.write(data)
        target = cv2.resize(np.moveaxis((visible / 8).astype(np.uint8), 0, -1), self.target_size)
        Image.fromarray(target).save(self.paths["rgb"])
        self.pair = discover_pairs(self.root)[0]

    def pan_result(self, accepted=True):
        return {"accepted": accepted, "H_native_to_target": self.h.tolist(),
                "metrics": {"inliers": 100, "inlier_ratio": 1.0, "holdout_count": 40,
                            "source_coverage": 0.8, "target_coverage": 0.8,
                            "holdout_median_msi_px": 0.01, "holdout_p90_msi_px": 0.1,
                            "holdout_inlier_fraction": 1.0},
                "strategy": "synthetic_pan", "attempts": [],
                "reason": None if accepted else "Synthetic strict PAN QA rejection"}

    @staticmethod
    def bridge_result(accepted):
        return {"accepted": accepted, "H_pan_to_rgb": np.eye(3).tolist(),
                "metrics": {"holdout_median_msi_px": 0.01, "holdout_p90_msi_px": 0.1,
                            "source_coverage": 0.8, "target_coverage": 0.8},
                "reason": None if accepted else "Synthetic bridge QA rejection"}

    def source_snapshot(self):
        return {name: (path.stat().st_size, path.stat().st_mtime_ns,
                       hashlib.sha256(path.read_bytes()).hexdigest()) for name, path in self.paths.items()}

    def product_from(self, record):
        for name in ("product", "rgb_product", "pan_product"):
            if isinstance(record.get(name), dict) and "nir" in record[name]:
                return record[name]
        if "nir" in record:
            return record
        self.fail(f"No persisted product in record: {record}")

    def assert_product(self, product, grid):
        nir_path, valid_path = Path(product["nir"]), Path(product["valid"])
        self.assertEqual(nir_path, self.output / "JAX_100" / "JAX_100_13_nir.tif")
        self.assertEqual(valid_path, self.output / "JAX_100" / "JAX_100_13_valid.png")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(nir_path) as dataset:
                values = dataset.read(1)
                valid = dataset.read_masks(1) > 0
                tags = dataset.tags()
                self.assertEqual(dataset.count, 1)
                self.assertEqual(dataset.dtypes, ("float32",))
                self.assertTrue(np.isnan(dataset.nodata))
                self.assertEqual((dataset.width, dataset.height), self.target_size)
                self.assertFalse(any(path.endswith(".msk") for path in dataset.files))
        with Image.open(valid_path) as image:
            png = np.asarray(image)
        self.assertTrue(set(np.unique(png)).issubset({0, 255}))
        np.testing.assert_array_equal(valid, png > 0)
        self.assertTrue(np.isnan(values[~valid]).all())
        self.assertTrue(np.isfinite(values[valid]).all())
        self.assertGreater(float(values[valid].min()), 4000)
        self.assertLessEqual(float(values[valid].max()), float(self.nir.max()))
        self.assertEqual(tags.get("CURRENT_RGB_READY", "").lower(), "true" if grid == "rgb" else "false")
        self.assertIn(grid.upper(), tags.get("REFERENCE_GRID", tags.get("GRID", "")).upper())
        self.assertFalse(list(self.output.rglob("*.tmp")))

    def test_both_grids_use_original_aoi_and_main_product_names(self):
        pan_paths = recovery.product_paths(self.pair, self.output, "pan")
        rgb_paths = recovery.product_paths(self.pair, self.output, "rgb")
        self.assertEqual(pan_paths, rgb_paths)
        self.assertEqual(pan_paths, (self.output / "JAX_100" / "JAX_100_13_nir.tif",
                                     self.output / "JAX_100" / "JAX_100_13_valid.png"))

    def test_pan_only_product_preserves_dn_masks_and_false_rgb_tag(self):
        before = self.source_snapshot()
        product = recovery.save_product(self.nir, self.valid, self.h, self.target_size,
                                         self.pair, self.output, "pan", "config-test")
        self.assert_product(product, "pan")
        self.assertEqual(before, self.source_snapshot())

    def test_pan_success_and_bridge_rejection_save_pan_only(self):
        before = self.source_snapshot()
        with mock.patch.object(recovery, "estimate_pan_registration", return_value=self.pan_result()), \
                mock.patch.object(recovery, "estimate_rgb_bridge", return_value=self.bridge_result(False)):
            record = recovery.process_pair(self.pair, self.output, config_hash="config-test")
        self.assertEqual(record["status"], "pan_only", record)
        self.assertTrue(record["pan_accepted"])
        self.assertFalse(record["rgb_ready"])
        self.assertEqual(record["reference_grid"], "PAN")
        self.assert_product(self.product_from(record), "pan")
        self.assertEqual(before, self.source_snapshot())

    def test_verified_bridge_saves_single_final_rgb_warp(self):
        with mock.patch.object(recovery, "estimate_pan_registration", return_value=self.pan_result()), \
                mock.patch.object(recovery, "estimate_rgb_bridge", return_value=self.bridge_result(True)), \
                mock.patch.object(recovery, "save_product", wraps=recovery.save_product) as save:
            record = recovery.process_pair(self.pair, self.output, config_hash="config-test")
        self.assertEqual(record["status"], "rgb_ready", record)
        self.assertTrue(record["rgb_ready"])
        self.assertEqual(record["reference_grid"], "RGB")
        self.assertEqual(save.call_count, 1, "Do not resample PAN output again to RGB")
        self.assert_product(self.product_from(record), "rgb")
        self.assertEqual(len(list(self.output.rglob("*_nir.tif"))), 1)

    def test_rejected_pan_candidate_with_raw_h_never_creates_product(self):
        with mock.patch.object(recovery, "estimate_pan_registration", return_value=self.pan_result(False)), \
                mock.patch.object(recovery, "estimate_rgb_bridge") as bridge:
            record = recovery.process_pair(self.pair, self.output, config_hash="config-test")
        self.assertEqual(record["status"], "rejected", record)
        self.assertFalse(record["pan_accepted"])
        bridge.assert_not_called()
        self.assertFalse(list(self.output.rglob("*_nir.tif")))
        self.assertFalse(list(self.output.rglob("*_valid.png")))

    def test_claimed_acceptance_with_invalid_h_does_not_save(self):
        bad = self.pan_result()
        bad["H_native_to_target"] = np.full((3, 3), np.nan).tolist()
        with mock.patch.object(recovery, "estimate_pan_registration", return_value=bad), \
                mock.patch.object(recovery, "estimate_rgb_bridge", return_value=self.bridge_result(False)):
            record = recovery.process_pair(self.pair, self.output, config_hash="config-test")
        self.assertIn(record["status"], ("rejected", "error"), record)
        self.assertFalse(list(self.output.rglob("*_nir.tif")))

    def test_foreign_v1_tiff_and_mask_are_never_overwritten(self):
        nir_path, valid_path = recovery.product_paths(self.pair, self.output, "pan")
        nir_path.parent.mkdir(parents=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(nir_path, "w", driver="GTiff", width=256, height=256,
                               count=1, dtype="float32", transform=Affine.identity()) as dataset:
                dataset.write(np.full((256, 256), 9876, np.float32), 1)
                dataset.update_tags(PIPELINE_VERSION="seo-nir-rgb-grid-v1")
        Image.fromarray(np.full((256, 256), 255, np.uint8)).save(valid_path)
        before = (nir_path.read_bytes(), valid_path.read_bytes())
        with self.assertRaises((ValueError, RuntimeError, FileExistsError)):
            recovery.save_product(self.nir, self.valid, self.h, self.target_size,
                                    self.pair, self.output, "pan", "config-test")
        self.assertEqual(before, (nir_path.read_bytes(), valid_path.read_bytes()))

    def test_output_allows_requested_v1_directory_but_rejects_sources(self):
        recovery.validate_output(self.root, self.output)
        recovery.validate_output(self.root, self.root / "nir_pan_other")
        for relative in ("msi", "msi/child", "crops", "pansharpened_he", "shadows_max", "rpcs"):
            with self.subTest(relative=relative), self.assertRaises((ValueError, RuntimeError)):
                recovery.validate_output(self.root, self.root / relative)

    def test_same_version_product_from_different_config_is_protected(self):
        product = recovery.save_product(self.nir, self.valid, self.h, self.target_size,
                                         self.pair, self.output, "pan", "config-first")
        paths = [Path(product[field]) for field in ("nir", "valid")]
        before = [path.read_bytes() for path in paths]
        with self.assertRaises((ValueError, RuntimeError, FileExistsError)):
            recovery.save_product(self.nir, self.valid, self.h, self.target_size,
                                    self.pair, self.output, "rgb", "config-other")
        self.assertEqual(before, [path.read_bytes() for path in paths])

    def test_unpaired_foreign_mask_is_protected(self):
        nir_path, valid_path = recovery.product_paths(self.pair, self.output, "pan")
        valid_path.parent.mkdir(parents=True)
        Image.fromarray(np.full((256, 256), 255, np.uint8)).save(valid_path)
        before = valid_path.read_bytes()
        with self.assertRaises((ValueError, RuntimeError, FileExistsError)):
            recovery.save_product(self.nir, self.valid, self.h, self.target_size,
                                    self.pair, self.output, "pan", "config-test")
        self.assertEqual(before, valid_path.read_bytes())
        self.assertFalse(nir_path.exists())

    def test_invalid_bridge_h_falls_back_to_valid_pan_product(self):
        bridge = self.bridge_result(True)
        bridge["H_pan_to_rgb"] = np.full((3, 3), np.nan).tolist()
        with mock.patch.object(recovery, "estimate_pan_registration", return_value=self.pan_result()), \
                mock.patch.object(recovery, "estimate_rgb_bridge", return_value=bridge):
            record = recovery.process_pair(self.pair, self.output, config_hash="config-test")
        self.assertEqual(record["status"], "pan_only", record)
        self.assertFalse(record["rgb_ready"])
        self.assert_product(record["product"], "pan")

    def test_composition_validation_exception_falls_back_to_pan(self):
        pan = self.pan_result()
        yy, xx = np.mgrid[8:57:8, 8:57:8]
        source_points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
        pan["_source_points"] = source_points
        pan["_pan_points"] = cv2.perspectiveTransform(source_points[:, None], self.h)[:, 0]
        with mock.patch.object(recovery, "estimate_pan_registration", return_value=pan), \
                mock.patch.object(recovery, "estimate_rgb_bridge", return_value=self.bridge_result(True)), \
                mock.patch.object(recovery, "check_composition", side_effect=ValueError("synthetic invalid composition")):
            record = recovery.process_pair(self.pair, self.output, config_hash="config-test")
        self.assertEqual(record["status"], "pan_only", record)
        self.assertFalse(record["rgb_ready"])
        self.assert_product(record["product"], "pan")

    def test_preview_failure_does_not_erase_saved_pan_status(self):
        with mock.patch.object(recovery, "estimate_pan_registration", return_value=self.pan_result()), \
                mock.patch.object(recovery, "estimate_rgb_bridge", return_value=self.bridge_result(False)), \
                mock.patch.object(Image.Image, "save", side_effect=OSError("synthetic preview failure")):
            record = recovery.process_pair(self.pair, self.output, preview=True, config_hash="config-test")
        self.assertEqual(record["status"], "pan_only", record)
        self.assertIn("preview_error", record)
        self.assert_product(record["product"], "pan")

    def test_matcher_runtime_failures_are_retryable_errors(self):
        matcher = mock.Mock()
        matcher.match.side_effect = RuntimeError("synthetic CUDA OOM")
        with mock.patch.object(recovery, "match_full_resolution", side_effect=RuntimeError("synthetic extractor failure")):
            record = recovery.process_pair(self.pair, self.output, matcher=matcher, config_hash="config-test")
        self.assertEqual(record["status"], "error", record)
        self.assertFalse(recovery.reusable(record, self.pair, self.output, "config-test"))
        self.assertFalse(list(self.output.rglob("*_nir.tif")))

    def test_resume_checks_config_files_and_transient_bridge_errors(self):
        with mock.patch.object(recovery, "estimate_pan_registration", return_value=self.pan_result()), \
                mock.patch.object(recovery, "estimate_rgb_bridge", return_value=self.bridge_result(False)):
            record = recovery.process_pair(self.pair, self.output, config_hash="config-test")
        self.assertTrue(recovery.reusable(record, self.pair, self.output, "config-test"))
        self.assertFalse(recovery.reusable(record, self.pair, self.output, "different-config"))
        retry = {**record, "rgb_bridge": {"accepted": False, "execution_error": True}}
        self.assertFalse(recovery.reusable(retry, self.pair, self.output, "config-test"))
        Path(record["product"]["valid"]).unlink()
        self.assertFalse(recovery.reusable(record, self.pair, self.output, "config-test"))

    def test_selection_includes_all_705_failures_not_only_638_matching_failures(self):
        old_key = self.pair["key"]
        originals = {old_key: {"status": "ok"}}
        pairs = [self.pair]
        failed_keys = set()
        for index in range(705):
            key = f"OMA_{index}/OMA_{index}_12"
            failed_keys.add(key)
            originals[key] = {"status": "failed", "error": "Only 2 mutual matches" if index < 638 else "Holdout mismatch"}
            pairs.append({**self.pair, "key": key})
        with mock.patch.object(recovery, "discover_pairs", return_value=pairs):
            selected = recovery.select_remaining_pairs(self.root, originals)
            self.assertEqual({pair["key"] for pair in selected}, failed_keys)
            self.assertEqual(len(selected), 705)
            with self.assertRaises(ValueError):
                recovery.select_remaining_pairs(self.root, originals, keys={old_key})

    def test_mixed_csv_training_manifest_excludes_pan_only_and_rejections(self):
        self.output.mkdir(parents=True)
        originals = {"old": {"status": "ok", "nir": "old-nir", "valid": "old-valid", "valid_fraction": 0.99},
                     "pan": {"status": "failed"}, "rgb": {"status": "failed"}, "reject": {"status": "failed"}}
        recovered = {
            "pan": {"status": "pan_only", "reference_grid": "PAN", "rgb_ready": False,
                    "product": {"nir": "pan-nir", "valid": "pan-valid", "current_rgb_ready": False}},
            "rgb": {"status": "rgb_ready", "reference_grid": "RGB", "rgb_ready": True,
                    "product": {"nir": "rgb-nir", "valid": "rgb-valid", "current_rgb_ready": True}},
            "reject": {"status": "rejected", "rgb_ready": False},
        }
        recovery.write_tables(self.output, originals, recovered, {"pan", "rgb", "reject"})
        with (self.output / "registration_all.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["key"] for row in rows}, set(originals))
        pan_row = next(row for row in rows if row["key"] == "pan")
        self.assertEqual(pan_row["reference_grid"], "PAN")
        self.assertEqual(pan_row["current_rgb_ready"].lower(), "false")
        with (self.output / "rgb_training_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
            training = list(csv.DictReader(handle))
        self.assertEqual({row["key"] for row in training}, {"old", "rgb"})

    def test_cli_recovers_truncated_manifest_resumes_and_preserves_v1_history(self):
        self.output.mkdir(parents=True)
        legacy_nir = self.output / "JAX_99" / "JAX_99_1_nir.tif"
        legacy_valid = legacy_nir.with_name("JAX_99_1_valid.png")
        legacy_nir.parent.mkdir()
        legacy_nir.write_bytes(b"legacy NIR must not change")
        legacy_valid.write_bytes(b"legacy mask must not change")
        original_records = [{"key": "JAX_99/JAX_99_1", "status": "ok", "nir": str(legacy_nir),
                             "valid": str(legacy_valid), "valid_fraction": 0.99},
                            {"key": self.pair["key"], "status": "failed", "error": "Holdout mismatch"}]
        old_manifest = self.output / "registration.jsonl"
        old_config = self.output / "processing_config.json"
        old_manifest.write_text("".join(json.dumps(record) + "\n" for record in original_records), encoding="utf-8")
        old_config.write_text('{"version":"original-v1"}\n', encoding="utf-8")
        protected = {path: path.read_bytes() for path in (legacy_nir, legacy_valid, old_manifest, old_config)}
        new_manifest = self.output / (recovery.PREFIX + "_registration.jsonl")
        new_manifest.write_text('{"key":"interrupted', encoding="utf-8")
        submitted = []

        class ImmediateExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, function, pair, *arguments):
                submitted.append(pair["key"])
                future = Future()
                future.set_result({"key": pair["key"], "status": "rejected", "pan_accepted": False,
                                   "rgb_ready": False, "version": recovery.VERSION,
                                   "inputs": recovery.input_fingerprint(pair), "config_hash": arguments[-1],
                                   "reason": "Synthetic strict QA rejection"})
                return future

        arguments = ["recover_seo_nir_pan.py", "--root", str(self.root), "--output", str(self.output), "--no-loftr"]
        with mock.patch.object(sys, "argv", arguments), mock.patch.object(recovery, "ProcessPoolExecutor", ImmediateExecutor), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recovery.main(), 0)
            self.assertEqual(submitted, [self.pair["key"]])
            submitted.clear()
            self.assertEqual(recovery.main(), 0)
        self.assertEqual(submitted, [])
        self.assertEqual(set(recovery.load_records(new_manifest)), {self.pair["key"]})
        self.assertEqual(protected, {path: path.read_bytes() for path in protected})
        summary = json.loads((self.output / (recovery.PREFIX + "_summary.json")).read_text())
        self.assertEqual(summary["selected"], 1)
        self.assertEqual(summary["attempted_this_run"], 0)


if __name__ == "__main__":
    unittest.main()
