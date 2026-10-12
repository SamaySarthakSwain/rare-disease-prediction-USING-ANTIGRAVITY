import unittest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from inference import predict_rare_diseases

class TestInferencePipeline(unittest.TestCase):
    def test_empty_input(self):
        res = predict_rare_diseases([])
        self.assertEqual(res["status"], "INSUFFICIENT_INFORMATION")
        self.assertEqual(len(res["ranked_candidates"]), 0)

    def test_invalid_identifiers(self):
        res = predict_rare_diseases(["INVALID:0000000", "UNKNOWN_TERM"])
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("INVALID:0000000", res["unknown_terms"])

    def test_valid_clinical_query(self):
        # HP:0001250 (Seizure), HP:0001249 (Intellectual disability)
        res = predict_rare_diseases(["HP:0001250", "HP:0001249"], top_k=5)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(len(res["differential_diagnosis"]), 5)
        
        top1 = res["differential_diagnosis"][0]
        self.assertIn("disease_id", top1)
        self.assertIn("disease_name", top1)
        self.assertIn("ranking_score", top1)
        self.assertGreater(top1["ranking_score"], 0.0)
        self.assertIn("directly_matched_phenotypes", top1)
        self.assertIn("sample_unreported_disease_phenotypes", top1)
        self.assertIn("disclaimer", res)

    def test_sparse_query_warning(self):
        # Single phenotype query
        res = predict_rare_diseases(["HP:0001250"], top_k=3)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(any("Sparse query" in w for w in res["warnings"]))

if __name__ == "__main__":
    unittest.main()
