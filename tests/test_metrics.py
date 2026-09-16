import math
import unittest

import numpy as np

from metrics import case_metrics, summarize


class TestSegmentationMetrics(unittest.TestCase):
    def test_overlap_metrics_and_distances(self):
        truth = np.zeros((8, 8, 8), dtype=np.uint8); truth[2:5, 2:5, 2:5] = 1
        pred = truth.copy()
        row = case_metrics(pred, truth, (1.0, 1.0, 1.0))
        self.assertEqual(row["tp"], 27)
        self.assertEqual(row["fp"], 0)
        self.assertEqual(row["fn"], 0)
        for key in ("dice", "iou", "ppv", "sen"):
            self.assertEqual(row[key], 1.0)
        self.assertEqual(row["hd95_mm"], 0.0)
        self.assertEqual(row["asd_mm"], 0.0)

    def test_empty_prediction_is_reported_strictly(self):
        truth = np.zeros((8, 8, 8), dtype=np.uint8); truth[2:5, 2:5, 2:5] = 1
        row = case_metrics(np.zeros_like(truth), truth, (1.0, 1.0, 1.0))
        self.assertEqual(row["dice"], 0.0); self.assertEqual(row["iou"], 0.0)
        self.assertEqual(row["ppv"], 0.0); self.assertEqual(row["sen"], 0.0)
        self.assertTrue(math.isinf(row["hd95_mm"])); self.assertTrue(math.isinf(row["asd_mm"]))
        summary = summarize([row])
        self.assertTrue(math.isinf(summary["hd95_strict_mean_mm"]))
        self.assertTrue(math.isinf(summary["asd_strict_mean_mm"]))
        self.assertEqual(summary["empty_prediction_rate"], 1.0)

    def test_empty_target_and_prediction_is_correct(self):
        empty = np.zeros((8, 8, 8), dtype=np.uint8)
        row = case_metrics(empty, empty, (1.0, 1.0, 1.0))
        for key in ("dice", "iou", "ppv", "sen"):
            self.assertEqual(row[key], 1.0)
        self.assertEqual(row["hd95_mm"], 0.0); self.assertEqual(row["asd_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
