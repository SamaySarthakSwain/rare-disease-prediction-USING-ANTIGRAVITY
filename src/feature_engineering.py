"""
Feature Engineering and Semantic Similarity Computation for Rare-Disease-AI
Fast, cached Information Content (IC), Most Informative Common Ancestor (MICA),
Resnik and Lin semantic similarities, and vectorized Best-Match Average (BMA).
Strictly prevents test leakage by fitting IC solely on training disease annotations.
"""

import math
from collections import defaultdict

class SemanticSimilarityEngine:
    def __init__(self, ontology, training_disease_phenotypes):
        """
        ontology: HPOntology instance
        training_disease_phenotypes: dict of disease_id -> set of active HPO terms
        """
        self.ontology = ontology
        self.disease_phenotypes = training_disease_phenotypes
        self.total_diseases = max(1, len(training_disease_phenotypes))
        self.ic = {}               # term_id -> float
        self.term_disease_freq = defaultdict(int)
        self.mica_cache = {}       # (t1, t2) -> mica_term
        self.sim_cache = {}        # (t1, t2, metric) -> float
        self._fit_information_content()
        
    def _fit_information_content(self):
        """
        Compute IC = -ln(freq(t) / N) where freq(t) is count of diseases annotated with t
        or any descendant of t (i.e. propagating annotations up to ancestors).
        """
        for did, terms in self.disease_phenotypes.items():
            all_ancestors = set()
            for t in terms:
                all_ancestors.update(self.ontology.get_ancestors(t))
            for anc in all_ancestors:
                self.term_disease_freq[anc] += 1
                
        for term_id in self.ontology.names.keys():
            freq = self.term_disease_freq.get(term_id, 0)
            prob = (freq + 1) / (self.total_diseases + 1)
            self.ic[term_id] = -math.log(prob)
            
    def get_ic(self, term_id):
        canonical = self.ontology.resolve_term(term_id)
        if not canonical:
            return 0.0
        return self.ic.get(canonical, 0.0)

    def find_mica(self, t1, t2):
        """Finds Most Informative Common Ancestor of two HPO terms."""
        if t1 == t2:
            return t1
        key = (min(t1, t2), max(t1, t2))
        if key in self.mica_cache:
            return self.mica_cache[key]
            
        anc1 = self.ontology.get_ancestors(t1)
        anc2 = self.ontology.get_ancestors(t2)
        common = anc1.intersection(anc2)
        
        if not common:
            self.mica_cache[key] = None
            return None
            
        best_term = max(common, key=lambda t: self.ic.get(t, 0.0))
        self.mica_cache[key] = best_term
        return best_term

    def resnik_similarity(self, t1, t2):
        if t1 == t2:
            return self.get_ic(t1)
        key = (min(t1, t2), max(t1, t2), "resnik")
        if key in self.sim_cache:
            return self.sim_cache[key]
        mica = self.find_mica(t1, t2)
        sim = self.get_ic(mica) if mica else 0.0
        self.sim_cache[key] = sim
        return sim

    def lin_similarity(self, t1, t2):
        if t1 == t2:
            return 1.0
        key = (min(t1, t2), max(t1, t2), "lin")
        if key in self.sim_cache:
            return self.sim_cache[key]
        mica = self.find_mica(t1, t2)
        if not mica:
            self.sim_cache[key] = 0.0
            return 0.0
        ic_mica = self.get_ic(mica)
        ic1 = self.get_ic(t1)
        ic2 = self.get_ic(t2)
        denom = ic1 + ic2
        sim = min(1.0, (2.0 * ic_mica) / denom) if denom > 1e-9 else 0.0
        self.sim_cache[key] = sim
        return sim

    def score_patient_against_disease(self, patient_terms, disease_terms, metric="resnik"):
        """
        Fast Patient-Recall oriented similarity:
        Average of max similarities for each patient phenotype across the disease phenotypes.
        """
        if not patient_terms or not disease_terms:
            return 0.0
        sim_fn = self.resnik_similarity if metric == "resnik" else self.lin_similarity
        
        total = 0.0
        for p in patient_terms:
            best_s = 0.0
            # Shortcut: if exact match is present, it is the theoretical maximum!
            if p in disease_terms:
                best_s = self.get_ic(p) if metric == "resnik" else 1.0
            else:
                for d in disease_terms:
                    s = sim_fn(p, d)
                    if s > best_s:
                        best_s = s
            total += best_s
            
        return total / len(patient_terms)
