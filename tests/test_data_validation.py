import unittest
import os
import json

class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.files = [
            r"Dataset 1/hp.obo",
            r"Dataset 2/phenotype (1).hpoa",
            r"Dataset 3/genes_to_disease.txt",
            r"Dataset 4/genes_to_phenotype.txt"
        ]
        self.inventory_path = "reports/data_inventory.json"
        self.quality_path = "reports/data_quality_report.json"

    def test_datasets_exist(self):
        for f in self.files:
            self.assertTrue(os.path.exists(f), f"Dataset file missing: {f}")
            self.assertGreater(os.path.getsize(f), 0, f"Dataset file empty: {f}")

    def test_inventory_report_generated(self):
        self.assertTrue(os.path.exists(self.inventory_path))
        with open(self.inventory_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            self.assertIn("datasets", data)
            self.assertEqual(len(data["datasets"]), 4)

    def test_quality_report_metrics(self):
        self.assertTrue(os.path.exists(self.quality_path))
        with open(self.quality_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            obo = data["obo_summary"]
            self.assertGreater(obo["active_terms"], 19000)
            self.assertEqual(obo["cycles_found"], 0, "Ontology must not have cycles")
            
            hpoa = data["hpoa_summary"]
            self.assertGreater(hpoa["total_rows"], 200000)
            self.assertGreater(hpoa["unique_diseases"], 10000)
            self.assertGreater(hpoa["qualifier_counts"].get("NOT", 0), 0)

            g2d = data["g2d_summary"]
            self.assertGreater(g2d["total_rows"], 10000)
            
            g2p = data["g2p_summary"]
            self.assertGreater(g2p["total_rows"], 300000)

if __name__ == "__main__":
    unittest.main()
