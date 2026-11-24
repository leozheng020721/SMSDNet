"""Candidate persistence checks on synthetic temporary data only."""

import contextlib
from concurrent.futures import Future
import csv
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import warnings

import numpy as np
from PIL import Image
import rasterio
from rasterio.errors import NotGeoreferencedWarning

import recover_seo_nir_candidates as candidate
import test_recover_seo_nir_pan as fixtures


class CandidateProductTests(unittest.TestCase):
    setUp = fixtures.RecoveryProductTests.setUp
    source_snapshot = fixtures.RecoveryProductTests.source_snapshot
    pan_result = fixtures.RecoveryProductTests.pan_result
    bridge_result = staticmethod(fixtures.RecoveryProductTests.bridge_result)

    def process(self, bridge, fitted=None, **kwargs):
        with mock.patch.object(candidate, "estimate_candidate_registration",
                               return_value=fitted if fitted is not None else self.pan_result()), \
                mock.patch.object(candidate, "estimate_rgb_bridge", return_value=bridge):
            return candidate.process_pair(self.pair, self.output, config_hash="test", **kwargs)

    def assert_candidate(self, record, grid):
        self.assertEqual(record["status"], "candidate_" + grid, record)
        for value in (record, record["product"]):
            self.assertIs(value["registration_verified"], False)
            self.assertIs(value["current_rgb_ready"], False)
            self.assertEqual(value["quality_status"], candidate.QUALITY)
        product = record["product"]
        nir_path, valid_path = candidate.product_paths(self.pair, self.output)
        self.assertEqual(Path(product["nir"]), nir_path)
        self.assertEqual(Path(product["valid"]), valid_path)
        self.assertEqual(nir_path, self.output / "JAX_100" / "JAX_100_13_nir.tif")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(nir_path) as dataset:
                values, valid, tags = dataset.read(1), dataset.read_masks(1) > 0, dataset.tags()
                self.assertEqual(dataset.dtypes, ("float32",))
                self.assertTrue(np.isnan(dataset.nodata))
                self.assertFalse(any(path.endswith(".msk") for path in dataset.files))
        with Image.open(valid_path) as image:
            png = np.asarray(image)
        np.testing.assert_array_equal(valid, png > 0)
        self.assertTrue(set(np.unique(png)) <= {0, 255})
        self.assertTrue(np.isnan(values[~valid]).all())
        self.assertTrue(np.isfinite(values[valid]).all())
        self.assertGreater(float(values[valid].min()), 4000)
        self.assertLessEqual(float(values[valid].max()), float(self.nir.max()))
        self.assertEqual(tags["PIPELINE_VERSION"], candidate.VERSION)
        self.assertEqual(tags["REGISTRATION_STATUS"], candidate.QUALITY)
        self.assertEqual(tags["REGISTRATION_VERIFIED"], "false")
        self.assertEqual(tags["CURRENT_RGB_READY"], "false")
        self.assertEqual(tags["REFERENCE_GRID"], grid.upper())
        self.assertEqual(tags["UNITS"], "raw_DN")
        self.assertFalse(list(self.output.rglob("*.tmp")))

    def test_rgb_bridge_success_still_saves_unverified_raw_dn_and_masks(self):
        before = self.source_snapshot()
        with mock.patch.object(candidate, "save_product", wraps=candidate.save_product) as save:
            record = self.process(self.bridge_result(True))
        self.assert_candidate(record, "rgb")
        self.assertIs(record["product"]["rgb_grid_verified"], True)
        self.assertEqual(save.call_count, 1)
        self.assertEqual(before, self.source_snapshot())

    def test_bridge_rejection_saves_pan_candidate(self):
        record = self.process(self.bridge_result(False))
        self.assert_candidate(record, "pan")
        self.assertIs(record["product"]["rgb_grid_verified"], False)

    def test_invalid_composition_falls_back_to_pan(self):
        bridge = self.bridge_result(True)
        bridge["H_pan_to_rgb"] = np.full((3, 3), np.nan).tolist()
        record = self.process(bridge)
        self.assert_candidate(record, "pan")
        self.assertIn("composition_reason", record)

    def test_rejected_or_invalid_mapping_never_fabricates_output(self):
        for accepted in (False, True):
            with self.subTest(accepted=accepted):
                fitted = self.pan_result(accepted)
                fitted["H_native_to_target"] = None
                record = self.process(self.bridge_result(False), fitted=fitted)
                self.assertIn(record["status"], ("rejected", "error"))
                self.assertFalse(list(self.output.rglob("*_nir.tif")))
                self.assertFalse(list(self.output.rglob("*_valid.png")))

    def test_foreign_versions_or_config_are_not_overwritten(self):
        before_sources = self.source_snapshot()
        cases = (("seo-nir-rgb-grid-v1", "test"), ("seo-nir-pan-recovery-v2", "test"),
                 (candidate.VERSION, "different-config"))
        for index, (version, requested_config) in enumerate(cases):
            with self.subTest(version=version, requested_config=requested_config):
                output = self.output / str(index)
                product = candidate.save_product(self.nir, self.valid, self.h, self.target_size,
                                                 self.pair, output, "pan", "test")
                paths = [Path(product[name]) for name in ("nir", "valid")]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NotGeoreferencedWarning)
                    with rasterio.open(paths[0], "r+") as dataset:
                        dataset.update_tags(PIPELINE_VERSION=version)
                before = [path.read_bytes() for path in paths]
                with self.assertRaises(FileExistsError):
                    candidate.save_product(self.nir, self.valid, self.h, self.target_size,
                                           self.pair, output, "rgb", requested_config)
                self.assertEqual(before, [path.read_bytes() for path in paths])
        self.assertEqual(before_sources, self.source_snapshot())

    def test_orphan_validity_mask_is_protected(self):
        nir_path, valid_path = candidate.product_paths(self.pair, self.output)
        valid_path.parent.mkdir(parents=True)
        Image.fromarray(np.full((256, 256), 255, np.uint8)).save(valid_path)
        before = valid_path.read_bytes()
        with self.assertRaises(FileExistsError):
            candidate.save_product(self.nir, self.valid, self.h, self.target_size,
                                   self.pair, self.output, "pan", "test")
        self.assertEqual(before, valid_path.read_bytes())
        self.assertFalse(nir_path.exists())

    def test_swallowed_bridge_oom_saves_pan_but_remains_retryable(self):
        bridge = {"accepted": False, "attempts": [{"strategy": "grid",
                  "reason": "OpenCV(-4: Insufficient memory) in cv::OutOfMemoryError"}]}
        record = self.process(bridge)
        self.assert_candidate(record, "pan")
        self.assertTrue(record["rgb_bridge"].get("execution_error"))
        self.assertFalse(candidate.reusable(record, self.pair, self.output, "test"))
        summary = candidate.summarize({record["key"]: record}, {record["key"]})
        self.assertEqual(summary["retry_keys"], [record["key"]])
        self.assertEqual(summary["saved"], 1)
        self.assertEqual(summary["errors"], 0)

    def test_reusable_checks_configuration_and_can_repair_own_mask(self):
        record = self.process(self.bridge_result(False))
        self.assertTrue(candidate.reusable(record, self.pair, self.output, "test"))
        self.assertFalse(candidate.reusable(record, self.pair, self.output, "other"))
        Path(record["product"]["valid"]).unlink()
        self.assertFalse(candidate.reusable(record, self.pair, self.output, "test"))
        repaired = self.process(self.bridge_result(False))
        self.assert_candidate(repaired, "pan")
        self.assertTrue(candidate.reusable(repaired, self.pair, self.output, "test"))

    def test_baseline_excludes_selected_and_detects_ledger_and_product_changes(self):
        self.process(self.bridge_result(False))
        old_nir = self.output / "JAX_99" / "JAX_99_1_nir.tif"
        old_nir.parent.mkdir()
        old_nir.write_bytes(b"protected old NIR")
        old_valid = old_nir.with_name("JAX_99_1_valid.png")
        old_valid.write_bytes(b"protected old mask")
        ledger = self.output / "rgb_training_manifest.csv"
        ledger.write_bytes(b"protected training ledger")
        baseline = candidate.capture_baseline(self.output, {self.pair["key"]})
        self.assertEqual(baseline["protected_nir_count"], 1)
        self.assertEqual(baseline["protected_valid_count"], 1)
        self.assertEqual(candidate.check_baseline(self.output, baseline), [])
        ledger.write_bytes(b"changed training ledger")
        self.assertTrue(any("ledger" in issue.lower() for issue in candidate.check_baseline(self.output, baseline)))
        ledger.write_bytes(b"protected training ledger")
        old_nir.write_bytes(b"changed protected old NIR with a different size")
        self.assertTrue(any("product" in issue.lower() for issue in candidate.check_baseline(self.output, baseline)))

    def test_selection_keeps_all_688_rejected_and_protects_17_strict_successes(self):
        old_key = self.pair["key"]
        originals, strict = {old_key: {"status": "ok"}}, {}
        pairs, rejected, protected = [self.pair], set(), {old_key}
        for index in range(705):
            key = f"OMA_{index}/OMA_{index}_12"
            originals[key] = {"status": "failed"}
            strict[key] = {"status": "rejected" if index < 688 else
                           ("rgb_ready" if index % 2 else "pan_only")}
            pairs.append({**self.pair, "key": key})
            (rejected if index < 688 else protected).add(key)
        with mock.patch.object(candidate, "discover_pairs", return_value=pairs):
            selected = candidate.select_remaining_pairs(self.root, originals, strict)
            self.assertEqual({pair["key"] for pair in selected}, rejected)
            self.assertEqual(len(selected), 688)
            self.assertEqual(len(protected - {old_key}), 17)
            for key in protected:
                with self.subTest(protected_key=key), self.assertRaises(ValueError):
                    candidate.select_remaining_pairs(self.root, originals, strict, keys=[key])

    def test_candidate_csv_is_separate_and_does_not_modify_training_ledger(self):
        self.output.mkdir(parents=True)
        training = self.output / "rgb_training_manifest.csv"
        old_all = self.output / "registration_all.csv"
        training.write_bytes(b"existing verified RGB training list\n")
        old_all.write_bytes(b"existing strict index\n")
        protected = {path: path.read_bytes() for path in (training, old_all)}
        originals = {"old": {"status": "ok"}, "strict-rgb": {"status": "failed"},
                     "strict-pan": {"status": "failed"}, "rgb": {"status": "failed"},
                     "pan": {"status": "failed"}, "reject": {"status": "failed"}}
        strict = {"strict-rgb": {"status": "rgb_ready", "reference_grid": "RGB"},
                  "strict-pan": {"status": "pan_only", "reference_grid": "PAN"}}
        records = {key: {"key": key, "status": "candidate_" + key,
                         "product": {"reference_grid": key.upper(), "registration_verified": False,
                                     "current_rgb_ready": False, "quality_status": candidate.QUALITY}}
                   for key in ("rgb", "pan")}
        records["reject"] = {"key": "reject", "status": "rejected"}
        candidate.write_tables(self.output, originals, strict, records, set(records))
        with (self.output / "registration_with_candidates.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = {row["key"]: row for row in csv.DictReader(handle)}
        self.assertEqual(set(rows), set(originals))
        for key in records:
            self.assertEqual(rows[key]["registration_verified"].lower(), "false")
            self.assertEqual(rows[key]["current_rgb_ready"].lower(), "false")
        with (self.output / "rgb_candidate_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual({row["key"] for row in csv.DictReader(handle)}, {"rgb"})
        self.assertEqual(protected, {path: path.read_bytes() for path in protected})

    def test_cli_resume_reuses_tuple_config_and_preserves_history_and_product(self):
        self.output.mkdir(parents=True)
        original = self.output / "registration.jsonl"
        strict = self.output / "pan_recovery_registration.jsonl"
        training = self.output / "rgb_training_manifest.csv"
        original.write_text(json.dumps({"key": self.pair["key"], "status": "failed"}) + "\n", encoding="utf-8")
        strict.write_text(json.dumps({"key": self.pair["key"], "status": "rejected"}) + "\n", encoding="utf-8")
        training.write_bytes(b"verified training list stays unchanged\n")
        protected = {path: path.read_bytes() for path in (original, strict, training)}
        submitted = []

        class ImmediateExecutor:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, function, pair, *arguments):
                submitted.append(pair["key"])
                future = Future()
                try:
                    future.set_result(function(pair, *arguments))
                except Exception as exc:
                    future.set_exception(exc)
                return future

        args = ["recover_seo_nir_candidates.py", "--root", str(self.root),
                "--output", str(self.output), "--workers", "1", "--no-loftr"]
        with mock.patch.object(sys, "argv", args), \
                mock.patch.object(candidate, "IsolatedImageExecutor", ImmediateExecutor), \
                mock.patch.object(candidate, "estimate_candidate_registration", return_value=self.pan_result()), \
                mock.patch.object(candidate, "estimate_rgb_bridge", return_value=self.bridge_result(True)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(candidate.main(), 0)
            self.assertEqual(submitted, [self.pair["key"]])
            product_paths = candidate.product_paths(self.pair, self.output)
            before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in product_paths}
            submitted.clear()
            self.assertEqual(candidate.main(), 0)
        self.assertEqual(submitted, [])
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in product_paths})
        self.assertEqual(protected, {path: path.read_bytes() for path in protected})
        summary = json.loads((self.output / "pan_candidate_summary.json").read_text())
        self.assertEqual(summary["attempted_this_run"], 0)
        self.assertTrue(summary["historical_integrity_ok"])


if __name__ == "__main__":
    unittest.main()
