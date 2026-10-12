import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from metrics import compute_ranking_metrics, compute_confidence_intervals

class TestMetrics(unittest.TestCase):
    def test_ranking_metrics_perfect(self):
        preds = [
            {"true_id": "D1", "ranked_candidates": ["D1", "D2", "D3"]},
            {"true_id": "D2", "ranked_candidates": ["D2", "D1", "D3"]}
        ]
        res = compute_ranking_metrics(preds)
        self.assertEqual(res["hits@1"], 1.0)
        self.assertEqual(res["mrr"], 1.0)
        self.assertEqual(res["hits@5"], 1.0)

    def test_ranking_metrics_partial(self):
        preds = [
            {"true_id": "D1", "ranked_candidates": ["D2", "D1", "D3"]}, # rank 2 (RR=0.5)
            {"true_id": "D2", "ranked_candidates": ["D9", "D8", "D2"]}  # rank 3 (RR=0.3333)
        ]
        res = compute_ranking_metrics(preds)
        self.assertEqual(res["hits@1"], 0.0)
        self.assertEqual(res["hits@5"], 1.0)
        self.assertAlmostEqual(res["mrr"], (0.5 + 1/3) / 2, places=3)

    def test_bootstrap_ci(self):
        preds = [{"true_id": "D1", "ranked_candidates": ["D1", "D2"]}] * 20
        ci = compute_confidence_intervals(preds, n_bootstraps=50)
        self.assertEqual(ci["hits@1_95ci"], (1.0, 1.0))

if __name__ == "__main__":
    unittest.main()
