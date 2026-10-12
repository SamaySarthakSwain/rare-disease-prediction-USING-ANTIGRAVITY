import unittest
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from ontology import HPOntology

class TestIdentifierMapping(unittest.TestCase):
    def setUp(self):
        self.mapping_path = "processed/identifier_mapping.csv"
        self.meta_path = "processed/disease_metadata.csv"
        self.dp_path = "processed/disease_phenotype.csv"
        self.gd_path = "processed/gene_disease_processed.csv"

    def test_mappings_exist(self):
        self.assertTrue(os.path.exists(self.mapping_path))
        df = pd.read_csv(self.mapping_path)
        self.assertGreater(len(df), 0)
        self.assertIn("canonical_identifier", df.columns)
        self.assertIn("mapping_status", df.columns)

    def test_gene_canonicalization(self):
        df_gd = pd.read_csv(self.gd_path)
        # All ncbi_gene_ids must start with NCBI:
        self.assertTrue(df_gd["ncbi_gene_id"].str.startswith("NCBI:").all())

    def test_disease_metadata_schema(self):
        self.assertTrue(os.path.exists(self.meta_path))
        df_meta = pd.read_csv(self.meta_path)
        self.assertIn("disease_id", df_meta.columns)
        self.assertIn("num_affirmative_phenotypes", df_meta.columns)
        self.assertIn("num_associated_genes", df_meta.columns)
        self.assertGreater(len(df_meta), 10000)

    def test_no_not_in_affirmative_qualifier(self):
        df_dp = pd.read_csv(self.dp_path)
        self.assertIn("qualifier", df_dp.columns)
        qualifiers = set(df_dp["qualifier"].unique())
        self.assertTrue({"AFFIRMATIVE", "NOT"}.issubset(qualifiers) or qualifiers == {"AFFIRMATIVE", "NOT"})

if __name__ == "__main__":
    unittest.main()
