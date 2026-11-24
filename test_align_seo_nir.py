"""CPU checks for NIR pixel geometry, persisted products, and restart behavior."""

import json
import os
from pathlib import Path
import subprocess
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

from align_seo_nir import (
    AlignmentError,
    discover_pairs,
    estimate_registration,
    input_fingerprint,
    output_paths,
    process_pair,
    resize_coordinate_matrix,
    reusable,
    stretch_rgb,
    validate_homography,
    warp_nir,
)


def textured_rgb(size=192):
    rng = np.random.default_rng(1234)
    image = rng.integers(30, 220, (size, size, 3), dtype=np.uint8)
    for _ in range(150):
        point = tuple(int(x) for x in rng.integers(5, size - 5, 2))
        color = tuple(int(x) for x in rng.integers(0, 256, 3))
        cv2.circle(image, point, int(rng.integers(2, 7)), color, -1)
    return image


class GeometryTests(unittest.TestCase):
    def test_resize_maps_pixel_centers_including_anisotropic_half_pixels(self):
        matrix = resize_coordinate_matrix((3, 2), (12, 6))
        points = np.array([[[0, 0], [2, 1], [-0.5, -0.5], [2.5, 1.5]]], np.float64)
        mapped = cv2.perspectiveTransform(points, matrix)[0]
        np.testing.assert_allclose(mapped, [[1.5, 1], [9.5, 4], [-0.5, -0.5], [11.5, 5.5]])

    def test_resize_rejects_nonpositive_dimensions(self):
        with self.assertRaises(ValueError):
            resize_coordinate_matrix((0, 2), (4, 4))

    def test_constant_raw_dn_survives_one_pass_resampling(self):
        raw = np.full((8, 10), 4097, np.uint16)
        matrix = resize_coordinate_matrix((10, 8), (20, 24))
        aligned, valid = warp_nir(raw, np.ones_like(raw, bool), matrix, (20, 24))
        self.assertEqual(aligned.dtype, np.float32)
        self.assertEqual(valid.dtype, np.bool_)
        self.assertEqual(aligned.shape, (24, 20))
        self.assertGreater(valid.mean(), 0.7)
        np.testing.assert_allclose(aligned[valid], 4097)
        self.assertTrue(np.isnan(aligned[~valid]).all())

    def test_partial_interpolation_support_is_invalid(self):
        raw = np.full((9, 9), 777, np.float32)
        raw[3, 3] = 1e9
        source_valid = np.ones_like(raw, bool)
        source_valid[3, 3] = False
        matrix = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], np.float64)
        aligned, valid = warp_nir(raw, source_valid, matrix, (9, 9))
        self.assertFalse(valid[3:5, 3:5].any())
        np.testing.assert_allclose(aligned[valid], 777)

    def test_translation_excludes_outside_footprint(self):
        raw = np.arange(80, dtype=np.float32).reshape(8, 10)
        matrix = np.array([[1, 0, 3], [0, 1, 0], [0, 0, 1]], np.float64)
        aligned, valid = warp_nir(raw, np.ones_like(raw, bool), matrix, (10, 8))
        self.assertFalse(valid[:, :3].any())
        self.assertTrue(valid[:, 3:].all())
        np.testing.assert_array_equal(aligned[:, 3:], raw[:, :-3])
        self.assertTrue(np.isnan(aligned[:, :3]).all())

    def test_unusable_homographies_are_rejected(self):
        candidates = {
            "nonfinite": np.full((3, 3), np.nan),
            "singular": np.diag([1, 0, 1]),
            "reflection": np.array([[-1, 0, 19], [0, 1, 0], [0, 0, 1]]),
            "extreme_scale": np.diag([5, 5, 1]),
            "horizon": np.array([[1, 0, 0], [0, 1, 0], [-0.1, 0, 1]]),
            "outside": np.array([[1, 0, 100], [0, 1, 0], [0, 0, 1]]),
        }
        for name, matrix in candidates.items():
            with self.subTest(name=name), self.assertRaises(AlignmentError):
                validate_homography(matrix, (20, 20), (20, 20))

    def test_estimator_recovers_a_known_native_to_target_mapping(self):
        source = textured_rgb()
        expected = resize_coordinate_matrix((192, 192), (384, 384))
        expected[0, 2] += 3
        expected[1, 2] += 2
        target = cv2.warpPerspective(source, expected, (384, 384), flags=cv2.INTER_LINEAR)
        result = estimate_registration(source, target)
        points = np.float64([[[20, 20], [170, 20], [170, 170], [20, 170], [96, 96]]])
        predicted = cv2.perspectiveTransform(points, np.array(result["H_native_to_target"]))
        truth = cv2.perspectiveTransform(points, expected)
        self.assertLess(float(np.linalg.norm(predicted - truth, axis=2).max()), 1.0)
        self.assertGreaterEqual(result["holdout_count"], 4)

    def test_textureless_registration_does_not_silently_fall_back_to_resize(self):
        blank = np.full((64, 64, 3), 100, np.uint8)
        with self.assertRaises(AlignmentError):
            estimate_registration(blank, cv2.resize(blank, (128, 128)))


class PersistentProductTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "dataset"
        self.output = self.root / "nir_aligned_v1"
        self.stem = "JAX_100_13"
        self.source_rgb = textured_rgb()
        self.raw = np.moveaxis(self.source_rgb[:, :, ::-1], -1, 0).astype(np.uint16) * 4 + 1000
        nir = (np.indices((192, 192)).sum(axis=0) + 2000).astype(np.uint16)
        self.raw = np.concatenate((self.raw, nir[None]))
        self.target = cv2.resize(stretch_rgb(np.moveaxis(self.raw[[2, 1, 0]], 0, -1)),
                                 (384, 384), interpolation=cv2.INTER_LINEAR)
        paths = {
            "msi": self.root / "msi" / "JAX_100" / (self.stem + "_msi.tif"),
            "pan": self.root / "crops" / "JAX_100" / (self.stem + "_pan.tif"),
            "rgb": self.root / "pansharpened_he" / "JAX_100" / (self.stem + "_rgb.png"),
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            for name, data in (("msi", self.raw), ("pan", np.ones((1, 384, 384), np.uint16))):
                with rasterio.open(paths[name], "w", driver="GTiff", width=data.shape[2],
                                   height=data.shape[1], count=len(data), dtype="uint16",
                                   transform=Affine.translation(500, 900), crs="EPSG:32632") as ds:
                    ds.write(data)
        Image.fromarray(self.target).save(paths["rgb"])
        self.pair = discover_pairs(self.root)[0]

    def fixed_registration(self):
        return {"H_native_to_target": resize_coordinate_matrix((192, 192), (384, 384)).tolist(),
                "strategy": "test", "holdout_median_rgb_px": 0.0}

    def test_output_preserves_aoi_and_filename_and_validity(self):
        before = input_fingerprint(self.pair)
        with mock.patch("align_seo_nir.estimate_registration", return_value=self.fixed_registration()):
            record = process_pair(self.pair, self.output)
        self.assertEqual(record["status"], "ok", record.get("error"))
        nir_path, valid_path = output_paths(self.pair, self.output)
        self.assertEqual(nir_path.relative_to(self.output).as_posix(), "JAX_100/JAX_100_13_nir.tif")
        self.assertEqual(valid_path.relative_to(self.output).as_posix(), "JAX_100/JAX_100_13_valid.png")
        with rasterio.open(nir_path) as ds:
            values, valid = ds.read(1), ds.read_masks(1) > 0
            self.assertEqual(ds.count, 1)
            self.assertEqual(ds.dtypes, ("float32",))
            self.assertIsNone(ds.crs)
            self.assertTrue(np.isnan(ds.nodata))
            self.assertEqual(ds.tags()["UNITS"], "raw_DN")
        np.testing.assert_array_equal(np.array(Image.open(valid_path)) > 0, valid)
        self.assertTrue(np.isnan(values[~valid]).all())
        self.assertGreater(float(values[valid].min()), 1000)
        self.assertLessEqual(float(values[valid].max()), float(self.raw[3].max()))
        self.assertEqual(before, input_fingerprint(self.pair))
        self.assertFalse(list(self.output.rglob("*.tmp")))
        self.assertTrue(reusable(record, self.pair, self.output))
        source = Path(self.pair["msi"])
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000))
        self.assertFalse(reusable(record, self.pair, self.output))

    def run_cli(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("align_seo_nir.py")),
                                 "--root", str(self.root), "--workers", "1"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]

    def test_cli_resume_keeps_completed_raster_untouched(self):
        first = self.run_cli()
        self.assertEqual(first[-1]["accepted"], 1)
        nir_path, valid_path = output_paths(self.pair, self.output)
        before = (nir_path.stat().st_mtime_ns, valid_path.stat().st_mtime_ns)
        second = self.run_cli()
        self.assertEqual(second[0]["resume_skipped"], 1)
        self.assertEqual(second[0]["jobs"], 0)
        self.assertEqual(before, (nir_path.stat().st_mtime_ns, valid_path.stat().st_mtime_ns))

    def test_cli_recovers_after_manifest_is_interrupted_mid_line(self):
        self.output.mkdir(parents=True)
        manifest = self.output / "registration.jsonl"
        manifest.write_text('{"key":"interrupted', encoding="utf-8")
        self.run_cli()
        second = self.run_cli()
        self.assertEqual(second[0]["resume_skipped"], 1)
        self.assertEqual(second[0]["jobs"], 0)

    def test_cli_rejects_output_inside_original_data(self):
        output = self.root / "msi" / "generated"
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("align_seo_nir.py")),
                                 "--root", str(self.root), "--output", str(output)],
                                capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("separate from original", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
