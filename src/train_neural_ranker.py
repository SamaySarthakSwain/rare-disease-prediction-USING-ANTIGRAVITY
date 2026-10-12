"""
Training and Evaluation Pipeline for BiomedicalDeepNeuralRanker
Trains the deep neural ranker using ontology-grounded embeddings and listwise cross-entropy loss.
Evaluates on Setting 1 (Unseen Patients) and Setting 2 (Unseen Diseases Zero-Shot).
Saves the optimized model to models/best_graph_model/deep_neural_ranker.pt
"""

import os
import sys
import json
import csv
import time
import random
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ontology import HPOntology
from feature_engineering import SemanticSimilarityEngine
from graph_models import OntologyEmbeddingPretrainer, BiomedicalDeepNeuralRanker, ListwiseRankingLoss
from metrics import compute_ranking_metrics, compute_confidence_intervals

def log_msg(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def train_and_eval_neural_model():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    os.makedirs("models/best_graph_model", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Load Data
    log_msg("Loading HPO ontology and disease data...")
    hpo = HPOntology("Dataset 1/hp.obo")
    
    disease_phenos = defaultdict(set)
    disease_neg = defaultdict(set)
    disease_names = {}
    with open("processed/disease_phenotype.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            did = r["disease_id"]
            hid = r["hpo_id"]
            disease_names[did] = r["disease_name"]
            if r["qualifier"] == "AFFIRMATIVE":
                disease_phenos[did].add(hid)
            elif r["qualifier"] == "NOT":
                disease_neg[did].add(hid)

    disease_genes = defaultdict(set)
    with open("processed/gene_disease_processed.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            disease_genes[r["disease_id"]].add(r["ncbi_gene_id"])

    with open("processed/splits/disease_split_manifest.json", "r", encoding="utf-8") as f:
        split_meta = json.load(f)
    with open("processed/splits/evaluation_cases.json", "r", encoding="utf-8") as f:
        cases_data = json.load(f)

    train_diseases = set(split_meta["train_diseases"])
    val_diseases = set(split_meta["val_diseases"])
    test_diseases = set(split_meta["test_diseases"])

    # 2. Fit Semantic Engine strictly on training partition
    log_msg("Fitting Semantic Similarity Engine on training diseases...")
    train_profiles = {d: disease_phenos[d] for d in train_diseases if d in disease_phenos}
    semantic_engine = SemanticSimilarityEngine(hpo, train_profiles)

    # 3. Pre-train Ontology Embeddings via SVD
    log_msg("Computing ontology-grounded dense phenotype representations via DAG SVD...")
    dense_embeds, term2idx = OntologyEmbeddingPretrainer.fit_embeddings(hpo, semantic_engine, embed_dim=64)
    log_msg(f"Constructed dense embeddings for {dense_embeds.shape[0]} HPO terms.")

    # 4. Feature Extraction Function
    def extract_features(patient_terms, candidate_did, q_dense=None):
        d_aff = disease_phenos.get(candidate_did, set())
        d_neg = disease_neg.get(candidate_did, set())
        
        # 0: Resnik BMA
        resnik_bma = semantic_engine.score_patient_against_disease(patient_terms, d_aff, metric="resnik")
        # 1: Lin BMA
        lin_bma = semantic_engine.score_patient_against_disease(patient_terms, d_aff, metric="lin")
        
        # 2: Dense cosine similarity
        d_p_indices = [term2idx[p] for p in d_aff if p in term2idx]
        if q_dense is not None and len(d_p_indices) > 0:
            z_d = dense_embeds[d_p_indices].mean(dim=0)
            z_d = z_d / (torch.norm(z_d) + 1e-8)
            dense_sim = torch.dot(q_dense, z_d).item()
        else:
            dense_sim = 0.0
            
        # 3: Exact overlap ratio
        p_set = set(patient_terms)
        inter = len(p_set.intersection(d_aff))
        overlap_ratio = inter / (len(p_set) + 1e-6)
        
        # 4: Precision ratio
        prec_ratio = inter / (len(d_aff) + 1e-6)
        
        # 5: Patient specificity
        ic_sum = sum(semantic_engine.get_ic(t) for t in patient_terms)
        spec = ic_sum / (len(p_set) + 1e-6)
        
        # 6: Contradiction penalty
        contradiction = 1.0 if d_neg and p_set.intersection(d_neg) else 0.0
        
        # 7: Gene evidence
        has_gene = 1.0 if len(disease_genes.get(candidate_did, set())) > 0 else 0.0
        
        return [resnik_bma, lin_bma, dense_sim, overlap_ratio, prec_ratio, spec, contradiction, has_gene]

    # 5. Initialize Model
    log_msg("Initializing BiomedicalDeepNeuralRanker...")
    model = BiomedicalDeepNeuralRanker(dense_embeds, in_features=8, hidden_dim=64)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = ListwiseRankingLoss(temperature=0.3)

    train_cases = cases_data["train_cases"][:250]
    train_d_list = sorted(list(train_diseases))
    epochs = 6
    num_negatives = 10

    log_msg(f"Training deep neural ranker for {epochs} epochs with listwise cross-entropy...")
    model.train()
    for epoch in range(1, epochs + 1):
        random.shuffle(train_cases)
        total_loss = 0.0
        step_count = 0
        
        for case in train_cases:
            true_d = case["true_disease_id"]
            p_terms = case["observed_phenotypes"]
            
            p_indices = [term2idx[t] for t in p_terms if t in term2idx]
            if not p_indices:
                continue
            q_dense = dense_embeds[p_indices].mean(dim=0)
            q_dense = q_dense / (torch.norm(q_dense) + 1e-8)
            
            # Select candidate set: [true_disease, neg_1, neg_2, ..., neg_k]
            cand_list = [true_d]
            while len(cand_list) <= num_negatives:
                neg = random.choice(train_d_list)
                if neg != true_d and neg not in cand_list:
                    cand_list.append(neg)
                    
            feats = [extract_features(p_terms, cd, q_dense) for cd in cand_list]
            feat_tensor = torch.tensor(feats, dtype=torch.float32)
            
            scores = model(feat_tensor)
            loss = criterion(scores, target_idx=0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step_count += 1
            
        avg_loss = total_loss / max(1, step_count)
        log_msg(f"Epoch {epoch}/{epochs} - Listwise Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "models/best_graph_model/deep_neural_ranker.pt")
    log_msg("Saved model checkpoint: models/best_graph_model/deep_neural_ranker.pt")

    # 6. Evaluation Function
    model.eval()
    def evaluate_model(cases, candidate_pool, is_zero_shot=False):
        predictions = []
        with torch.no_grad():
            for case in cases:
                p_terms = case["observed_phenotypes"]
                true_d = case["true_disease_id"]
                
                p_indices = [term2idx[t] for t in p_terms if t in term2idx]
                q_dense = dense_embeds[p_indices].mean(dim=0) if p_indices else None
                if q_dense is not None:
                    q_dense = q_dense / (torch.norm(q_dense) + 1e-8)
                    
                pool = candidate_pool if true_d in candidate_pool else candidate_pool + [true_d]
                
                # Extract features for all candidates
                feats = [extract_features(p_terms, cd, q_dense) for cd in pool]
                feat_tensor = torch.tensor(feats, dtype=torch.float32)
                scores = model(feat_tensor).tolist()
                
                cand_scores = list(zip(pool, scores))
                cand_scores.sort(key=lambda x: x[1], reverse=True)
                ranked_ids = [c[0] for c in cand_scores]
                
                predictions.append({
                    "true_id": true_d,
                    "ranked_candidates": ranked_ids
                })
                
        metrics = compute_ranking_metrics(predictions)
        metrics.update(compute_confidence_intervals(predictions))
        return metrics

    test_seen_cases = cases_data["test_seen_disease_cases"][:100]
    test_unseen_cases = cases_data["test_unseen_disease_cases"][:100]
    cand_pool_setting1 = list(train_diseases)[:150]
    cand_pool_setting2 = list(test_diseases)[:150]

    log_msg("Evaluating Deep Neural Ranker on Setting 1 (Unseen Patients)...")
    m1 = evaluate_model(test_seen_cases, cand_pool_setting1, is_zero_shot=False)
    log_msg(f"-> Deep Neural Ranker [Setting 1]: R@1={m1['recall@1']*100:.2f}%, R@5={m1['recall@5']*100:.2f}%, MRR={m1['mrr']:.4f}")

    log_msg("Evaluating Deep Neural Ranker on Setting 2 (Unseen Diseases Zero-Shot)...")
    m2 = evaluate_model(test_unseen_cases, cand_pool_setting2, is_zero_shot=True)
    log_msg(f"-> Deep Neural Ranker [Setting 2]: R@1={m2['recall@1']*100:.2f}%, R@5={m2['recall@5']*100:.2f}%, MRR={m2['mrr']:.4f}")

    # Update reports/model_comparison.md with Deep Neural Ranker
    with open("reports/model_comparison.md", "a", encoding="utf-8") as f:
        f.write(f"| **BiomedicalDeepNeuralRanker (Ours)** | Setting 1 (Unseen Patients) | **{m1['recall@1']*100:.2f}%** | **{m1['recall@5']*100:.2f}%** | **{m1['recall@10']*100:.2f}%** | **{m1['mrr']:.4f}** | {m1.get('mrr_95ci')} |\n")
        f.write(f"| **BiomedicalDeepNeuralRanker (Ours)** | Setting 2 (Unseen Diseases Zero-Shot) | **{m2['recall@1']*100:.2f}%** | **{m2['recall@5']*100:.2f}%** | **{m2['recall@10']*100:.2f}%** | **{m2['mrr']:.4f}** | {m2.get('mrr_95ci')} |\n")

    # Update experiment_log.csv
    with open("reports/experiment_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EXP-DeepNeuralRanker-S1",
            datetime.now().isoformat(),
            "ML Optimization",
            "BiomedicalDeepNeuralRanker on Setting 1",
            "SUCCESS",
            f"R@1={m1['recall@1']}, R@5={m1['recall@5']}, MRR={m1['mrr']}"
        ])
        writer.writerow([
            "EXP-DeepNeuralRanker-S2",
            datetime.now().isoformat(),
            "ML Optimization",
            "BiomedicalDeepNeuralRanker on Setting 2",
            "SUCCESS",
            f"R@1={m2['recall@1']}, R@5={m2['recall@5']}, MRR={m2['mrr']}"
        ])

    log_msg("Neural model optimization completed successfully!")

if __name__ == "__main__":
    train_and_eval_neural_model()
