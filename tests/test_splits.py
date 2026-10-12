import unittest
import os
import json

class TestSplitsAndLeakage(unittest.TestCase):
    def setUp(self):
        self.manifest_path = "processed/splits/disease_split_manifest.json"
        self.cases_path = "processed/splits/evaluation_cases.json"

    def test_splits_files_exist(self):
        self.assertTrue(os.path.exists(self.manifest_path))
        self.assertTrue(os.path.exists(self.cases_path))

    def test_zero_disease_leakage(self):
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        train_d = set(data["train_diseases"])
        val_d = set(data["val_diseases"])
        test_d = set(data["test_diseases"])

        # Check disjointness
        self.assertEqual(len(train_d.intersection(test_d)), 0, "Train and test diseases must be disjoint")
        self.assertEqual(len(train_d.intersection(val_d)), 0, "Train and val diseases must be disjoint")
        self.assertEqual(len(val_d.intersection(test_d)), 0, "Val and test diseases must be disjoint")

    def test_case_structure(self):
        with open(self.cases_path, "r", encoding="utf-8") as f:
            cases = json.load(f)
        self.assertIn("train_cases", cases)
        self.assertIn("test_seen_disease_cases", cases)
        self.assertIn("test_unseen_disease_cases", cases)
        
        sample = cases["test_unseen_disease_cases"][0]
        self.assertIn("case_id", sample)
        self.assertIn("true_disease_id", sample)
        self.assertIn("observed_phenotypes", sample)
        self.assertGreater(len(sample["observed_phenotypes"]), 0)

if __name__ == "__main__":
    unittest.main()
