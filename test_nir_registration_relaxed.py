"""Synthetic behavior checks for relaxed, explicitly unverified registration."""

from dataclasses import replace
import json
import unittest
from unittest import mock

import cv2
import numpy as np

from align_seo_nir import resize_coordinate_matrix
from nir_registration_relaxed import RelaxedOptions, fit_relaxed


def mapped(points, h):
    return cv2.perspectiveTransform(np.asarray(points, np.float64)[:, None], h)[:, 0]


class RelaxedRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)

    def setUp(self):
        self.rng = np.random.default_rng(16)
        self.source_size = (400, 400)
        self.target_size = (1617, 1543)
        self.points = self.rng.uniform(40, 350, (150, 2))

    def correspondences(self, work_h, points=None, noise=0.02):
        source = self.points if points is None else points
        working = mapped(source, np.asarray(work_h, np.float64))
        working += self.rng.normal(0, noise, working.shape)
        full = resize_coordinate_matrix(self.source_size, self.target_size)
        return mapped(working, full), full @ work_h

    def assert_json_safe(self, result):
        json.dumps(result, allow_nan=False)

    def test_partial_affine_preserves_native_centers_and_prefers_simple_model(self):
        angle = np.deg2rad(2)
        work = np.array([[np.cos(angle), -np.sin(angle), 4],
                         [np.sin(angle), np.cos(angle), -3], [0, 0, 1]])
        target, truth = self.correspondences(work)
        result = fit_relaxed(self.points, target, self.source_size, self.target_size)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["model"], "partial_affine")
        control = np.array([[30, 30], [365, 30], [30, 360], [360, 360]], np.float64)
        error = np.linalg.norm(mapped(control, np.array(result["H_native_to_target"]))
                               - mapped(control, truth), axis=1)
        self.assertLess(float(error.max()), 0.2)
        self.assertEqual(result["validation_status"], "UNVERIFIED")
        self.assertFalse(result["metrics"]["independently_validated"])
        self.assertEqual(result["metrics"]["residual_kind"], "TRAINING_RESIDUAL")
        self.assert_json_safe(result)

    def test_full_affine_can_recover_shear_without_needless_homography(self):
        work = np.array([[0.98, 0.08, 3], [-0.02, 1.02, -3], [0, 0, 1]])
        target, truth = self.correspondences(work)
        result = fit_relaxed(self.points, target, self.source_size, self.target_size)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["model"], "affine")
        error = np.linalg.norm(mapped(self.points, np.array(result["H_native_to_target"]))
                               - mapped(self.points, truth), axis=1)
        self.assertLess(float(np.median(error)), 0.2)
        self.assert_json_safe(result)

    def test_homography_requires_a_material_improvement(self):
        work = np.array([[1.0, 0.02, 4], [0.005, 0.98, 1], [0.0007, -0.0002, 1]])
        target, truth = self.correspondences(work)
        result = fit_relaxed(self.points, target, self.source_size, self.target_size)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["model"], "homography")
        error = np.linalg.norm(mapped(self.points, np.array(result["H_native_to_target"]))
                               - mapped(self.points, truth), axis=1)
        self.assertLess(float(np.median(error)), 0.2)
        self.assert_json_safe(result)

    def test_local_two_dimensional_support_is_retained_as_unverified(self):
        points = self.rng.uniform(90, 105, (16, 2))
        work = np.array([[1, 0, 1.5], [0, 1, 0.8], [0, 0, 1]], np.float64)
        target, _ = self.correspondences(work, points, noise=0)
        result = fit_relaxed(points, target, self.source_size, self.target_size)
        self.assertTrue(result["accepted"], result)
        self.assertLess(result["metrics"]["source_inlier_hull_fraction"], 0.01)
        self.assertFalse(result["metrics"]["coverage_threshold_required"])
        self.assertEqual(result["metrics"]["validation_status"], "UNVERIFIED")
        self.assert_json_safe(result)

    def test_majority_outliers_do_not_become_independent_validation(self):
        points = self.points[:100]
        work = np.array([[1, 0, 4], [0, 1, -2], [0, 0, 1]], np.float64)
        target, _ = self.correspondences(work, points)
        target[30:] = self.rng.uniform([0, 0], self.target_size, (70, 2))
        result = fit_relaxed(points, target, self.source_size, self.target_size)
        self.assertTrue(result["accepted"], result)
        self.assertGreaterEqual(result["metrics"]["inliers"], 25)
        self.assertLess(result["metrics"]["inlier_ratio"], 0.5)
        self.assertGreater(result["metrics"]["training_all_median_msi_px"], 20)
        self.assertLess(result["metrics"]["training_inlier_median_msi_px"], 0.2)
        self.assertFalse(result["metrics"]["independently_validated"])
        self.assert_json_safe(result)

    def test_duplicate_points_cannot_satisfy_minimum_match_count(self):
        points = np.repeat(self.points[:7], 4, axis=0)
        target, _ = self.correspondences(np.eye(3), points, noise=0)
        result = fit_relaxed(points, target, self.source_size, self.target_size)
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["H_native_to_target"])
        self.assertEqual(result["metrics"]["selected_matches"], 7)
        self.assertEqual(result["metrics"]["duplicate_matches_removed"], 21)
        self.assert_json_safe(result)

    def test_collinear_points_and_reflections_are_rejected(self):
        points = np.column_stack((np.linspace(30, 360, 30), np.full(30, 100)))
        target, _ = self.correspondences(np.eye(3), points, noise=0)
        line = fit_relaxed(points, target, self.source_size, self.target_size)
        self.assertFalse(line["accepted"])
        self.assertIn("collinear", line["reason"])
        work = np.array([[-1, 0, 399], [0, 1, 0], [0, 0, 1]], np.float64)
        target, _ = self.correspondences(work)
        reflection = fit_relaxed(self.points, target, self.source_size, self.target_size,
                                 options=replace(RelaxedOptions(), models=("affine", "homography")))
        self.assertFalse(reflection["accepted"])
        self.assertIsNone(reflection["H_native_to_target"])
        self.assert_json_safe(line)
        self.assert_json_safe(reflection)

    def test_tiny_geometric_overlap_is_rejected(self):
        points = np.float64([(x, y) for y in (1, 8, 15) for x in (40, 140, 240, 340)])
        work = np.array([[1, 0, 0], [0, 1, 380], [0, 0, 1]], np.float64)
        target, _ = self.correspondences(work, points, noise=0)
        result = fit_relaxed(points, target, self.source_size, self.target_size)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("overlap" in row["reason"] for row in result["candidates"]))
        self.assert_json_safe(result)

    def test_nonfinite_and_out_of_bounds_points_are_not_counted(self):
        points = self.points[:10].copy()
        target, _ = self.correspondences(np.eye(3), points, noise=0)
        points[0] = [np.nan, 2]
        target[1] = [-5, 4]
        result = fit_relaxed(points, target, self.source_size, self.target_size)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["metrics"]["selected_matches"], 8)
        self.assertEqual(result["metrics"]["nonfinite_matches"], 1)
        self.assertEqual(result["metrics"]["out_of_bounds_matches"], 1)
        self.assert_json_safe(result)

    def test_confidence_balancing_caps_dense_matches(self):
        points = self.rng.uniform(20, 360, (4000, 2))
        target, _ = self.correspondences(np.eye(3), points)
        result = fit_relaxed(points, target, self.source_size, self.target_size,
                             confidence=self.rng.random(len(points)),
                             options=replace(RelaxedOptions(), models=("partial_affine",)))
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["metrics"]["selected_matches"], 3000)
        self.assert_json_safe(result)

    def test_empty_and_malformed_inputs_reject_without_resize_fallback(self):
        for source, target in ((np.empty((0, 2)), np.empty((0, 2))),
                               (np.ones((8, 3)), np.ones((8, 3)))):
            with self.subTest(shape=source.shape):
                result = fit_relaxed(source, target, self.source_size, self.target_size)
                self.assertFalse(result["accepted"])
                self.assertIsNone(result["H_native_to_target"])
                self.assert_json_safe(result)

    def test_opencv_oom_propagates_from_model_and_outer_processing(self):
        target, _ = self.correspondences(np.eye(3))
        options = replace(RelaxedOptions(), models=("partial_affine",))
        for code, message in ((-4, "Insufficient resources"),
                               (-1, "Failed to allocate a buffer"),
                               (-1, "OutOfMemory")):
            error = cv2.error(message)
            error.code = code
            for mocked in ("nir_registration_relaxed.cv2.estimateAffinePartial2D",
                           "nir_registration_relaxed._balanced_subset"):
                with self.subTest(code=code, message=message, mocked=mocked):
                    with mock.patch(mocked, side_effect=error):
                        with self.assertRaises(cv2.error):
                            fit_relaxed(self.points, target, self.source_size,
                                        self.target_size, options=options)

    def test_python_memory_error_propagates_but_normal_opencv_error_rejects(self):
        target, _ = self.correspondences(np.eye(3))
        options = replace(RelaxedOptions(), models=("partial_affine",))
        with mock.patch("nir_registration_relaxed.cv2.estimateAffinePartial2D",
                        side_effect=MemoryError("Synthetic allocation failure")):
            with self.assertRaises(MemoryError):
                fit_relaxed(self.points, target, self.source_size, self.target_size, options=options)
        error = cv2.error("Bad argument")
        error.code = -5
        with mock.patch("nir_registration_relaxed.cv2.estimateAffinePartial2D", side_effect=error):
            result = fit_relaxed(self.points, target, self.source_size, self.target_size, options=options)
        self.assertFalse(result["accepted"])
        self.assertIn("Bad argument", result["candidates"][0]["reason"])
        self.assert_json_safe(result)


if __name__ == "__main__":
    unittest.main()
