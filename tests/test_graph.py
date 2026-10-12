import unittest
import os
import json
import pandas as pd

class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.nodes_path = "processed/knowledge_graph_nodes.csv"
        self.edges_path = "processed/knowledge_graph_edges.csv"
        self.meta_path = "processed/knowledge_graph_metadata.json"

    def test_graph_files_exist(self):
        self.assertTrue(os.path.exists(self.nodes_path))
        self.assertTrue(os.path.exists(self.edges_path))
        self.assertTrue(os.path.exists(self.meta_path))

    def test_metadata_counts(self):
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertGreater(meta["total_nodes"], 40000)
        self.assertGreater(meta["total_edges"], 500000)
        self.assertEqual(meta["cycles_in_is_a_hierarchy"], 0)
        self.assertIn("phenotype", meta["node_type_counts"])
        self.assertIn("disease", meta["node_type_counts"])
        self.assertIn("gene", meta["node_type_counts"])

    def test_edge_endpoint_integrity(self):
        df_nodes = pd.read_csv(self.nodes_path)
        node_ids = set(df_nodes["node_id"])
        
        # Sample first 10,000 edges for fast endpoint validation
        df_edges = pd.read_csv(self.edges_path, nrows=10000)
        self.assertTrue(df_edges["source"].isin(node_ids).all())
        self.assertTrue(df_edges["target"].isin(node_ids).all())

if __name__ == "__main__":
    unittest.main()
