"""Small CPU checks for the public single-image evaluation contract."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

from evaluate_registered_image import evaluate_binary, read_reference


class BinaryMetricTests(unittest.TestCase):
    def test_imbalanced_counts_ignore_invalid_pixels(self):
        # 3 positive and 7 negative labels; two extra pixels must not count.
        label = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]])
        pred = np.array([[1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1]])
        valid = np.ones_like(label, dtype=bool)
        valid[1, 4:] = False
        result = evaluate_binary(pred, label, valid)
        self.assertEqual({k: result[k] for k in ("TP", "TN", "FP", "FN", "N")},
                         {"TP": 2, "TN": 6, "FP": 1, "FN": 1, "N": 10})
        self.assertAlmostEqual(result["FNR"], 100 / 3)
        self.assertAlmostEqual(result["FPR"], 100 / 7)
        self.assertAlmostEqual(result["BER"], 500 / 21)
        self.assertAlmostEqual(result["shadowIoU"], 0.5)
        self.assertAlmostEqual(result["F1"], 2 / 3)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["OA"], 0.8)
        self.assertAlmostEqual(result["Kappa"], 11 / 21)
        self.assertAlmostEqual(result["shadow_prevalence"], 0.3)

    def test_absent_positive_class_does_not_report_perfect_ber(self):
        zeros = np.zeros((2, 3), dtype=np.uint8)
        result = evaluate_binary(zeros, zeros, np.ones_like(zeros, dtype=bool))
        for key in ("BER", "FNR", "recall", "precision", "shadowIoU", "F1", "Kappa"):
            self.assertIsNone(result[key], key)
        self.assertEqual(result["FPR"], 0.0)
        self.assertEqual(result["OA"], 1.0)

    def test_empty_or_mismatched_support_is_rejected(self):
        array = np.zeros((2, 3), dtype=bool)
        with self.assertRaisesRegex(ValueError, "no valid"):
            evaluate_binary(array, array, array)
        with self.assertRaisesRegex(ValueError, "identical 2D"):
            evaluate_binary(array, array.T, ~array)
        with self.assertRaisesRegex(ValueError, "identical 2D"):
            evaluate_binary(array.ravel(), array.ravel(), (~array).ravel())


class ReferenceReadingTests(unittest.TestCase):
    def test_literal_float_tiff_zero_survives_nodata_tag_and_unknowns_exclude(self):
        raw = np.array([[0, 255, 7], [np.nan, np.inf, -1]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.tif"
            metadata = TiffImagePlugin.ImageFileDirectory_v2()
            metadata[42113] = "0"
            Image.fromarray(raw).save(path, tiffinfo=metadata)
            label, known = read_reference(path, raw.shape)
            np.testing.assert_array_equal(label, [[False, True, False], [False, False, False]])
            np.testing.assert_array_equal(known, [[True, True, False], [False, False, False]])
            flipped_label, flipped_known = read_reference(path, raw.shape, "flip-ud")
            np.testing.assert_array_equal(flipped_label, [[False, False, False], [False, True, False]])
            np.testing.assert_array_equal(flipped_known, [[False, False, False], [True, True, False]])
            with self.assertRaisesRegex(ValueError, "no resizing"):
                read_reference(path, (3, 2))
            with self.assertRaisesRegex(ValueError, "orientation"):
                read_reference(path, raw.shape, "rotate-180")


if __name__ == "__main__":
    unittest.main()
