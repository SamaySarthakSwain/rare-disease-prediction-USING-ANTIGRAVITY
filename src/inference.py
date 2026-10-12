"""
Production-Ready Inference Pipeline for Rare-Disease-AI
Powered by BiomedicalDeepNeuralRanker and Semantic Similarity Engine.
Provides callable interface:
predict_rare_diseases(hpo_terms, patient_negative_terms=None, top_k=10)

Handles:
- Identifier validation & alias resolution
- Deep neural listwise reranking (BiomedicalDeepNeuralRanker)
- Insufficient information handling
- Ranked disease differential with matched phenotypes
- Unreported phenotype differentiation (clearly distinguishing unreported vs confirmed absent)
- Associated gene evidence and provenance
- Clinical decision-support disclaimer
"""

import os
import sys
import csv
import json
import math
import torch
from datetime import datetime
from collections import defaultdict

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ontology import HPOntology
from feature_engineering import SemanticSimilarityEngine
from baseline_models import WeightedOntologyModel
from graph_models import OntologyEmbeddingPretrainer, BiomedicalDeepNeuralRanker

class RareDiseasePredictor:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        self.version = "2.0.0-deep-neural-biomedical-ai"
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        obo_path = os.path.join(base_dir, "Dataset 1", "hp.obo")
        dp_path = os.path.join(base_dir, "processed", "disease_phenotype.csv")
        gd_path = os.path.join(base_dir, "processed", "gene_disease_processed.csv")
        meta_path = os.path.join(base_dir, "processed", "disease_metadata.csv")
        manifest_path = os.path.join(base_dir, "processed", "splits", "disease_split_manifest.json")
        model_path = os.path.join(base_dir, "models", "best_graph_model", "deep_neural_ranker.pt")
        
        self.ontology = HPOntology(obo_path)
        
        self.disease_phenos = defaultdict(set)
        self.disease_neg = defaultdict(set)
        self.disease_names = {}
        
        with open(dp_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                did = r["disease_id"]
                hid = r["hpo_id"]
                self.disease_names[did] = r["disease_name"]
                if r["qualifier"] == "AFFIRMATIVE":
                    self.disease_phenos[did].add(hid)
                elif r["qualifier"] == "NOT":
                    self.disease_neg[did].add(hid)
                    
        self.disease_genes = defaultdict(list)
        with open(gd_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.disease_genes[r["disease_id"]].append({
                    "ncbi_gene_id": r["ncbi_gene_id"],
                    "gene_symbol": r["gene_symbol"],
                    "association_type": r["association_type"],
                    "source": r["source"]
                })
                
        train_diseases = set()
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                train_diseases = set(manifest.get("train_diseases", []))
        else:
            train_diseases = set(self.disease_phenos.keys())
            
        train_profiles = {d: self.disease_phenos[d] for d in train_diseases if d in self.disease_phenos}
        self.semantic_engine = SemanticSimilarityEngine(self.ontology, train_profiles)
        self.baseline_model = WeightedOntologyModel(self.disease_phenos, self.semantic_engine, self.disease_neg)
        
        # Initialize Deep Neural Ranker
        self.use_neural = False
        try:
            self.dense_embeds, self.term2idx = OntologyEmbeddingPretrainer.fit_embeddings(
                self.ontology, self.semantic_engine, embed_dim=64
            )
            self.neural_model = BiomedicalDeepNeuralRanker(self.dense_embeds, in_features=8, hidden_dim=64)
            if os.path.exists(model_path):
                self.neural_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.neural_model.eval()
                self.use_neural = True
        except Exception as e:
            print(f"Warning: Falling back to ontology baseline ranker: {e}")
            self.use_neural = False

    def _extract_neural_features(self, patient_terms, candidate_did, q_dense):
        d_aff = self.disease_phenos.get(candidate_did, set())
        d_neg = self.disease_neg.get(candidate_did, set())
        
        resnik_bma = self.semantic_engine.score_patient_against_disease(patient_terms, d_aff, metric="resnik")
        lin_bma = self.semantic_engine.score_patient_against_disease(patient_terms, d_aff, metric="lin")
        
        d_p_indices = [self.term2idx[p] for p in d_aff if p in self.term2idx]
        if q_dense is not None and len(d_p_indices) > 0:
            z_d = self.dense_embeds[d_p_indices].mean(dim=0)
            z_d = z_d / (torch.norm(z_d) + 1e-8)
            dense_sim = torch.dot(q_dense, z_d).item()
        else:
            dense_sim = 0.0
            
        p_set = set(patient_terms)
        inter = len(p_set.intersection(d_aff))
        overlap_ratio = inter / (len(p_set) + 1e-6)
        prec_ratio = inter / (len(d_aff) + 1e-6)
        
        ic_sum = sum(self.semantic_engine.get_ic(t) for t in patient_terms)
        spec = ic_sum / (len(p_set) + 1e-6)
        contradiction = 1.0 if d_neg and p_set.intersection(d_neg) else 0.0
        has_gene = 1.0 if len(self.disease_genes.get(candidate_did, [])) > 0 else 0.0
        
        return [resnik_bma, lin_bma, dense_sim, overlap_ratio, prec_ratio, spec, contradiction, has_gene]

    def predict(self, hpo_terms, patient_negative_terms=None, top_k=10):
        timestamp = datetime.now().isoformat()
        
        if not hpo_terms or len(hpo_terms) == 0:
            return {
                "status": "INSUFFICIENT_INFORMATION",
                "message": "No phenotype identifiers were provided. At least one valid HPO term is required.",
                "original_input": hpo_terms,
                "ranked_candidates": [],
                "warnings": ["Empty phenotype query."],
                "disclaimer": "RESEARCH PROTOTYPE ONLY: This system is a biomedical AI decision-support experiment, not a clinically validated diagnostic device.",
                "timestamp": timestamp,
                "model_version": self.version
            }
            
        resolved_affirmative = []
        unknown_terms = []
        alias_resolutions = []
        
        for raw_t in hpo_terms:
            t_clean = raw_t.strip()
            resolved = self.ontology.resolve_term(t_clean)
            if not resolved:
                unknown_terms.append(t_clean)
            else:
                resolved_affirmative.append(resolved)
                if resolved != t_clean:
                    alias_resolutions.append({"original": t_clean, "resolved": resolved})
                    
        resolved_negative = []
        if patient_negative_terms:
            for raw_n in patient_negative_terms:
                n_clean = raw_n.strip()
                res = self.ontology.resolve_term(n_clean)
                if res:
                    resolved_negative.append(res)

        resolved_affirmative_set = set(resolved_affirmative)
        
        if not resolved_affirmative_set:
            return {
                "status": "INVALID_INPUT",
                "message": "None of the provided terms could be resolved to active HPO terms.",
                "original_input": hpo_terms,
                "unknown_terms": unknown_terms,
                "ranked_candidates": [],
                "warnings": ["All input terms failed ontology resolution."],
                "disclaimer": "RESEARCH PROTOTYPE ONLY: Not a clinically validated diagnostic device.",
                "timestamp": timestamp,
                "model_version": self.version
            }
            
        p_list = list(resolved_affirmative_set)
        
        # Fast two-stage candidate retrieval and neural reranking:
        # Stage 1: Retrieve top 200 candidates using fast semantic screening
        stage1_candidates = self.baseline_model.rank(p_list, patient_negative=resolved_negative, top_k=200)
        top_cand_ids = [c[0] for c in stage1_candidates]
        
        # Stage 2: Deep Neural Reranking
        if self.use_neural:
            p_indices = [self.term2idx[t] for t in p_list if t in self.term2idx]
            q_dense = self.dense_embeds[p_indices].mean(dim=0) if p_indices else None
            if q_dense is not None:
                q_dense = q_dense / (torch.norm(q_dense) + 1e-8)
                
            features = [self._extract_neural_features(p_list, did, q_dense) for did in top_cand_ids]
            feat_tensor = torch.tensor(features, dtype=torch.float32)
            with torch.no_grad():
                neural_scores = self.neural_model(feat_tensor).tolist()
            ranked_pairs = sorted(zip(top_cand_ids, neural_scores), key=lambda x: x[1], reverse=True)[:top_k]
        else:
            ranked_pairs = stage1_candidates[:top_k]
        
        differential = []
        for rank_idx, (did, score) in enumerate(ranked_pairs, start=1):
            dname = self.disease_names.get(did, "Unknown Disease Name")
            d_phenos = self.disease_phenos.get(did, set())
            
            exact_matches = resolved_affirmative_set.intersection(d_phenos)
            matched_details = []
            for m in sorted(exact_matches):
                matched_details.append({
                    "hpo_id": m,
                    "hpo_name": self.ontology.names.get(m, "Unknown"),
                    "ic": round(self.semantic_engine.get_ic(m), 4)
                })
                
            partial_matches = []
            for p in sorted(resolved_affirmative_set - exact_matches):
                best_d = max(d_phenos, key=lambda d: self.semantic_engine.resnik_similarity(p, d), default=None)
                if best_d:
                    sim = self.semantic_engine.resnik_similarity(p, best_d)
                    if sim > 1.0:
                        partial_matches.append({
                            "patient_hpo": p,
                            "patient_name": self.ontology.names.get(p, "Unknown"),
                            "closest_disease_hpo": best_d,
                            "disease_name": self.ontology.names.get(best_d, "Unknown"),
                            "resnik_ic": round(sim, 4)
                        })
                        
            unreported = list(d_phenos - resolved_affirmative_set)
            unreported_samples = [{
                "hpo_id": u,
                "hpo_name": self.ontology.names.get(u, "Unknown"),
                "status": "Unreported in patient profile (not confirmed absent)"
            } for u in sorted(unreported)[:5]]
            
            genes = self.disease_genes.get(did, [])
            
            if score > 8.0:
                interpretation = "STRONG_EVIDENCE"
            elif score > 4.0:
                interpretation = "MODERATE_EVIDENCE"
            elif score > 1.0:
                interpretation = "WEAK_EVIDENCE"
            else:
                interpretation = "INSUFFICIENT_EVIDENCE"
                
            differential.append({
                "rank": rank_idx,
                "disease_id": did,
                "disease_name": dname,
                "ranking_score": round(float(score), 4),
                "evidence_strength": interpretation,
                "directly_matched_phenotypes": matched_details,
                "semantically_related_phenotypes": partial_matches,
                "sample_unreported_disease_phenotypes": unreported_samples,
                "total_disease_phenotypes_count": len(d_phenos),
                "supporting_genes": genes[:5],
                "total_associated_genes_count": len(genes)
            })
            
        warnings = []
        if unknown_terms:
            warnings.append(f"Unrecognized terms omitted from ranking: {unknown_terms}")
        if len(resolved_affirmative_set) < 2:
            warnings.append("Sparse query (fewer than 2 valid phenotypes). Differential accuracy may be limited.")

        return {
            "status": "SUCCESS",
            "model_version": self.version,
            "prediction_timestamp": timestamp,
            "engine": "BiomedicalDeepNeuralRanker (Deep Neural Ranker + Ontology SVD Embeddings)" if self.use_neural else "WeightedOntologyModel",
            "query_summary": {
                "original_input_count": len(hpo_terms),
                "resolved_affirmative_count": len(resolved_affirmative_set),
                "resolved_affirmative_terms": sorted(list(resolved_affirmative_set)),
                "alias_resolutions": alias_resolutions,
                "unknown_terms": unknown_terms
            },
            "differential_diagnosis": differential,
            "warnings": warnings,
            "disclaimer": "RESEARCH PROTOTYPE ONLY: This system is a biomedical research prototype and clinical decision-support experiment, not a clinically validated diagnostic device. Predictions must not be represented as confirmed diagnoses."
        }

def predict_rare_diseases(hpo_terms, patient_negative_terms=None, top_k=10):
    predictor = RareDiseasePredictor.get_instance()
    return predictor.predict(hpo_terms, patient_negative_terms, top_k)
