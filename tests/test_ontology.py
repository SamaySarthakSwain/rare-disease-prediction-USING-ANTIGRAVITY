import unittest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from ontology import HPOntology

class TestOntology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hpo = HPOntology("Dataset 1/hp.obo")

    def test_term_resolution(self):
        # Test active term
        res = self.hpo.resolve_term("HP:0000118")
        self.assertEqual(res, "HP:0000118")

        # Test invalid term returns None
        self.assertIsNone(self.hpo.resolve_term("INVALID:9999999"))

    def test_ancestor_traversal(self):
        # HP:0001250 is Seizure, ancestor should include HP:0000118 (Phenotypic abnormality)
        anc = self.hpo.get_ancestors("HP:0001250")
        self.assertIn("HP:0001250", anc)
        self.assertIn("HP:0000118", anc)

    def test_no_infinite_loop_or_cycle(self):
        # Traversal should terminate
        for test_id in ["HP:0001250", "HP:0000924", "HP:0001627", "HP:0002086"]:
            anc = self.hpo.get_ancestors(test_id)
            self.assertGreater(len(anc), 1)

if __name__ == "__main__":
    unittest.main()
