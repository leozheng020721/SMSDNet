import math
import unittest

import numpy as np

from metrics import fast_hist, get_metrics


class SegmentationMetricsTest(unittest.TestCase):
    def test_binary_metrics(self):
        hist = np.array([[50, 10], [5, 35]])

        metrics = get_metrics(hist)

        self.assertAlmostEqual(metrics["ber"], 14.5833333333)
        self.assertAlmostEqual(metrics["iou"], 0.7)
        self.assertAlmostEqual(metrics["f1"], 70 / 85)
        self.assertAlmostEqual(metrics["kappa"], (0.85 - 0.51) / (1.0 - 0.51))

    def test_perfect_metrics(self):
        metrics = get_metrics(np.array([[8, 0], [0, 12]]))

        self.assertEqual(metrics["ber"], 0.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["kappa"], 1.0)

    def test_multiclass_uses_positive_class_one_vs_rest(self):
        hist = np.array(
            [
                [8, 1, 0],
                [2, 6, 1],
                [0, 1, 11],
            ]
        )

        metrics = get_metrics(hist, positive_class=1)

        self.assertAlmostEqual(metrics["ber"], 0.5 * (3 / 9 + 2 / 21) * 100)
        self.assertAlmostEqual(metrics["iou"], 6 / 11)
        self.assertAlmostEqual(metrics["f1"], 12 / 17)

    def test_fast_hist_ignores_invalid_labels(self):
        prediction = np.array([0, 1, 1, 2, 0])
        target = np.array([0, 1, 255, 1, -1])

        hist = fast_hist(prediction, target, num_classes=2)

        np.testing.assert_array_equal(hist, np.array([[1, 0], [0, 1]]))

    def test_empty_hist_returns_nan(self):
        metrics = get_metrics(np.zeros((2, 2)))

        self.assertTrue(all(math.isnan(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
