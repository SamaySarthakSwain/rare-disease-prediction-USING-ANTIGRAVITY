"""
Baseline Disease Ranking Models for Rare-Disease-AI
Models implemented:
1. Direct HPO Overlap Model
2. Jaccard Phenotype Similarity Model
3. Resnik Patient-Recall BMA Model
4. Lin Patient-Recall BMA Model
5. Weighted Ontology-Aware Model
"""

import math
from collections import defaultdict

class DirectOverlapModel:
    def __init__(self, disease_profiles):
        self.disease_profiles = disease_profiles
        
    def rank(self, patient_terms, candidate_diseases=None, top_k=10):
        p_set = set(patient_terms)
        candidates = candidate_diseases if candidate_diseases else list(self.disease_profiles.keys())
        scored = [(did, float(len(p_set.intersection(self.disease_profiles.get(did, set()))))) for did in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

class JaccardModel:
    def __init__(self, disease_profiles):
        self.disease_profiles = disease_profiles
        
    def rank(self, patient_terms, candidate_diseases=None, top_k=10):
        p_set = set(patient_terms)
        candidates = candidate_diseases if candidate_diseases else list(self.disease_profiles.keys())
        scored = []
        for did in candidates:
            d_set = self.disease_profiles.get(did, set())
            inter = len(p_set.intersection(d_set))
            union = len(p_set.union(d_set))
            score = float(inter / union) if union > 0 else 0.0
            scored.append((did, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

class ResnikBMAModel:
    def __init__(self, disease_profiles, semantic_engine):
        self.disease_profiles = disease_profiles
        self.semantic_engine = semantic_engine
        
    def rank(self, patient_terms, candidate_diseases=None, top_k=10):
        candidates = candidate_diseases if candidate_diseases else list(self.disease_profiles.keys())
        scored = []
        for did in candidates:
            score = self.semantic_engine.score_patient_against_disease(
                patient_terms, self.disease_profiles.get(did, set()), metric="resnik"
            )
            scored.append((did, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

class LinBMAModel:
    def __init__(self, disease_profiles, semantic_engine):
        self.disease_profiles = disease_profiles
        self.semantic_engine = semantic_engine
        
    def rank(self, patient_terms, candidate_diseases=None, top_k=10):
        candidates = candidate_diseases if candidate_diseases else list(self.disease_profiles.keys())
        scored = []
        for did in candidates:
            score = self.semantic_engine.score_patient_against_disease(
                patient_terms, self.disease_profiles.get(did, set()), metric="lin"
            )
            scored.append((did, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

class WeightedOntologyModel:
    def __init__(self, disease_profiles, semantic_engine, negative_profiles=None):
        self.disease_profiles = disease_profiles
        self.semantic_engine = semantic_engine
        self.negative_profiles = negative_profiles or defaultdict(set)
        
    def rank(self, patient_terms, patient_negative=None, candidate_diseases=None, top_k=10):
        candidates = candidate_diseases if candidate_diseases else list(self.disease_profiles.keys())
        p_set = set(patient_terms)
        ic_sum = sum(self.semantic_engine.get_ic(t) for t in patient_terms)
        norm_ic = ic_sum / (len(patient_terms) + 1e-6)
        spec_factor = 1.0 + 0.1 * math.log1p(norm_ic)
        
        scored = []
        for did in candidates:
            d_aff = self.disease_profiles.get(did, set())
            d_neg = self.negative_profiles.get(did, set())
            
            base_sim = self.semantic_engine.score_patient_against_disease(patient_terms, d_aff, metric="resnik")
            score = base_sim * spec_factor
            
            # Contradiction penalty
            if d_neg and p_set.intersection(d_neg):
                score *= 0.2
            if patient_negative and set(patient_negative).intersection(d_aff):
                score *= 0.5
                
            scored.append((did, score))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
