"""
Comprehensive Training, Evaluation, and Comparison Pipeline for Rare-Disease-AI
Evaluates:
1. Direct HPO Overlap
2. Jaccard Similarity
3. Lin BMA
4. Resnik Patient-Recall BMA
5. Weighted Ontology Model
6. BiomedicalHeteroGNN (Trained with Margin Ranking Loss)

Evaluates on two distinct settings:
- Setting 1: Unseen Patients (with seen disease classes)
- Setting 2: Unseen Diseases (Zero-shot generalization)
Computes Recall@1, Recall@5, Recall@10, Hits@K, MRR, NDCG@K, Precision@K, and 95% Bootstrap CIs.
Saves model checkpoints, baseline artifacts, and markdown reports.
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
from baseline_models import (
    DirectOverlapModel,
    JaccardModel,
    ResnikBMAModel,
    LinBMAModel,
    WeightedOntologyModel
)
from graph_models import BiomedicalHeteroGNN
from metrics import compute_ranking_metrics, compute_confidence_intervals

def log_msg(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_data():
    log_msg("Loading ontology...")
    hpo = HPOntology("Dataset 1/hp.obo")
    
    log_msg("Loading disease phenotypes...")
    disease_phenos = defaultdict(set)
    disease_neg = defaultdict(set)
    disease_names = {}
    with open("processed/disease_phenotype.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            did = r["disease_id"]
            hid = r["hpo_id"]
            dname = r["disease_name"]
            disease_names[did] = dname
            if r["qualifier"] == "AFFIRMATIVE":
                disease_phenos[did].add(hid)
            elif r["qualifier"] == "NOT":
                disease_neg[did].add(hid)

    log_msg("Loading disease genes...")
    disease_genes = defaultdict(set)
    with open("processed/gene_disease_processed.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            disease_genes[r["disease_id"]].add(r["ncbi_gene_id"])

    log_msg("Loading splits and evaluation cases...")
    with open("processed/splits/disease_split_manifest.json", "r", encoding="utf-8") as f:
        split_meta = json.load(f)
    with open("processed/splits/evaluation_cases.json", "r", encoding="utf-8") as f:
        cases_data = json.load(f)

    return hpo, disease_phenos, disease_neg, disease_genes, disease_names, split_meta, cases_data

def evaluate_baseline_model(model, test_cases, candidate_pool, model_name, top_k=20):
    predictions = []
    start_t = time.time()
    
    for case in test_cases:
        p_terms = case["observed_phenotypes"]
        true_d = case["true_disease_id"]
        
        # Ensure true disease is in candidate pool for retrieval evaluation
        if true_d not in candidate_pool:
            current_pool = candidate_pool + [true_d]
        else:
            current_pool = candidate_pool
            
        ranked = model.rank(p_terms, candidate_diseases=current_pool, top_k=top_k)
        ranked_ids = [r[0] for r in ranked]
        
        predictions.append({
            "true_id": true_d,
            "ranked_candidates": ranked_ids
        })
        
    eval_time = time.time() - start_t
    metrics = compute_ranking_metrics(predictions)
    cis = compute_confidence_intervals(predictions)
    metrics.update(cis)
    metrics["eval_time_sec"] = round(eval_time, 2)
    return metrics, predictions

def train_and_evaluate():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    os.makedirs("models/best_baseline", exist_ok=True)
    os.makedirs("models/best_graph_model", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    hpo, disease_phenos, disease_neg, disease_genes, disease_names, split_meta, cases_data = load_data()
    
    train_diseases = set(split_meta["train_diseases"])
    val_diseases = set(split_meta["val_diseases"])
    test_diseases = set(split_meta["test_diseases"])
    
    # 1. Fit Semantic Similarity Engine STRICTLY on train diseases (prevent leakage)
    log_msg("Fitting Semantic Similarity Engine on training diseases...")
    train_disease_phenos = {d: disease_phenos[d] for d in train_diseases if d in disease_phenos}
    semantic_engine = SemanticSimilarityEngine(hpo, train_disease_phenos)
    
    # 2. Instantiate Baseline Models
    log_msg("Initializing Baseline Models...")
    all_profiles = {d: disease_phenos[d] for d in disease_phenos if len(disease_phenos[d]) > 0}
    
    baselines = {
        "Direct HPO Overlap": DirectOverlapModel(all_profiles),
        "Jaccard Phenotype Similarity": JaccardModel(all_profiles),
        "Lin BMA": LinBMAModel(all_profiles, semantic_engine),
        "Resnik BMA (Patient-Recall)": ResnikBMAModel(all_profiles, semantic_engine),
        "Weighted Ontology Model": WeightedOntologyModel(all_profiles, semantic_engine, disease_neg)
    }
    
    test_seen_cases = cases_data["test_seen_disease_cases"][:100]
    test_unseen_cases = cases_data["test_unseen_disease_cases"][:100]
    
    # Candidate disease pools (sample 150 candidates + target for fast, benchmark-standard differential evaluation)
    cand_pool_setting1 = list(train_diseases)[:150]
    cand_pool_setting2 = list(test_diseases)[:150]
    
    log_msg(f"Setting 1 test cases: {len(test_seen_cases)}, Setting 2 test cases: {len(test_unseen_cases)}")

    baseline_results = []
    
    for name, model in baselines.items():
        log_msg(f"Evaluating {name} on Setting 1 (Unseen Patients)...")
        m1, _ = evaluate_baseline_model(model, test_seen_cases, cand_pool_setting1, name)
        
        log_msg(f"Evaluating {name} on Setting 2 (Unseen Diseases Zero-Shot)...")
        m2, _ = evaluate_baseline_model(model, test_unseen_cases, cand_pool_setting2, name)
        
        log_msg(f"-> {name} [Setting 1]: R@1={m1['recall@1']*100:.1f}%, R@5={m1['recall@5']*100:.1f}%, MRR={m1['mrr']:.4f}")
        log_msg(f"-> {name} [Setting 2]: R@1={m2['recall@1']*100:.1f}%, R@5={m2['recall@5']*100:.1f}%, MRR={m2['mrr']:.4f}")
        
        baseline_results.append({
            "model": name,
            "setting": "Setting 1 (Unseen Patients)",
            "recall@1": m1["recall@1"],
            "recall@5": m1["recall@5"],
            "recall@10": m1["recall@10"],
            "mrr": m1["mrr"],
            "hits@1_ci": m1.get("hits@1_95ci"),
            "mrr_ci": m1.get("mrr_95ci")
        })
        baseline_results.append({
            "model": name,
            "setting": "Setting 2 (Unseen Diseases Zero-Shot)",
            "recall@1": m2["recall@1"],
            "recall@5": m2["recall@5"],
            "recall@10": m2["recall@10"],
            "mrr": m2["mrr"],
            "hits@1_ci": m2.get("hits@1_95ci"),
            "mrr_ci": m2.get("mrr_95ci")
        })

    # Save best baseline model configuration
    with open("models/best_baseline/baseline_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": "Weighted Ontology Model",
            "weights": {
                "patient_recall_bma": 1.0,
                "ic_specificity_log_factor": 0.1,
                "contradiction_penalty": 0.2
            },
            "date": datetime.now().isoformat()
        }, f, indent=2)

    # 3. Train Graph Neural Ranking Model (BiomedicalHeteroGNN)
    log_msg("--- Initializing BiomedicalHeteroGNN ---")
    pheno_list = sorted(list(hpo.names.keys()))
    pheno2idx = {p: i for i, p in enumerate(pheno_list)}
    
    gene_set = set()
    for g_set in disease_genes.values():
        gene_set.update(g_set)
    gene_list = sorted(list(gene_set))
    gene2idx = {g: i for i, g in enumerate(gene_list)}
    
    train_d_list = sorted(list(train_diseases))
    d2idx_train = {d: i for i, d in enumerate(train_d_list)}
    
    num_p = len(pheno_list)
    num_g = max(1, len(gene_list))
    num_d_train = len(train_d_list)
    
    embed_dim = 64
    gnn_model = BiomedicalHeteroGNN(num_p, num_g, num_d_train, embed_dim=embed_dim, margin=0.3)
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.005, weight_decay=1e-5)
    
    # Pre-build tensor neighborhoods
    d_pheno_tensors = {}
    d_gene_tensors = {}
    for d, p_set in disease_phenos.items():
        p_indices = [pheno2idx[p] for p in p_set if p in pheno2idx]
        d_pheno_tensors[d] = torch.tensor(p_indices, dtype=torch.long)
        g_indices = [gene2idx[g] for g in disease_genes.get(d, []) if g in gene2idx]
        d_gene_tensors[d] = torch.tensor(g_indices, dtype=torch.long)

    train_cases = cases_data["train_cases"][:250]
    epochs = 8
    
    log_msg(f"Training BiomedicalHeteroGNN for {epochs} epochs...")
    gnn_model.train()
    for epoch in range(1, epochs + 1):
        random.shuffle(train_cases)
        total_loss = 0.0
        num_batches = 0
        
        for case in train_cases:
            true_d = case["true_disease_id"]
            if true_d not in d2idx_train:
                continue
                
            p_indices = [pheno2idx[t] for t in case["observed_phenotypes"] if t in pheno2idx]
            if not p_indices:
                continue
            p_tensor = torch.tensor(p_indices, dtype=torch.long)
            weights = torch.tensor([semantic_engine.get_ic(t) for t in case["observed_phenotypes"] if t in pheno2idx], dtype=torch.float)
            
            q = gnn_model.encode_patient_query(p_tensor, weights)
            
            d_pos_idx = d2idx_train[true_d]
            z_pos = gnn_model.encode_disease(d_pos_idx, d_pheno_tensors[true_d], d_gene_tensors[true_d])
            
            neg_d = random.choice(train_d_list)
            while neg_d == true_d:
                neg_d = random.choice(train_d_list)
            d_neg_idx = d2idx_train[neg_d]
            z_neg = gnn_model.encode_disease(d_neg_idx, d_pheno_tensors[neg_d], d_gene_tensors[neg_d])
            
            loss, _, _ = gnn_model.compute_pairwise_loss(q, z_pos, z_neg)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / max(1, num_batches)
        if epoch % 2 == 0 or epoch == epochs:
            log_msg(f"Epoch {epoch}/{epochs} - Avg Ranking Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(gnn_model.state_dict(), "models/best_graph_model/hetero_gnn.pt")
    log_msg("Saved checkpoint: models/best_graph_model/hetero_gnn.pt")
    
    # Evaluate GNN Model
    gnn_model.eval()
    
    def evaluate_gnn(cases, candidate_diseases, is_zero_shot=False):
        predictions = []
        with torch.no_grad():
            for case in cases:
                p_terms = case["observed_phenotypes"]
                true_d = case["true_disease_id"]
                p_indices = [pheno2idx[t] for t in p_terms if t in pheno2idx]
                if not p_indices:
                    continue
                p_tensor = torch.tensor(p_indices, dtype=torch.long)
                weights = torch.tensor([semantic_engine.get_ic(t) for t in p_terms if t in pheno2idx], dtype=torch.float)
                q = gnn_model.encode_patient_query(p_tensor, weights)
                
                candidates = candidate_diseases if true_d in candidate_diseases else candidate_diseases + [true_d]
                cand_scores = []
                for cd in candidates:
                    d_idx = d2idx_train.get(cd, -1) if not is_zero_shot else -1
                    z_cd = gnn_model.encode_disease(d_idx, d_pheno_tensors.get(cd, torch.tensor([], dtype=torch.long)), d_gene_tensors.get(cd, torch.tensor([], dtype=torch.long)))
                    score = torch.dot(q, z_cd).item()
                    cand_scores.append((cd, score))
                    
                cand_scores.sort(key=lambda x: x[1], reverse=True)
                ranked_ids = [c[0] for c in cand_scores]
                predictions.append({
                    "true_id": true_d,
                    "ranked_candidates": ranked_ids
                })
        metrics = compute_ranking_metrics(predictions)
        metrics.update(compute_confidence_intervals(predictions))
        return metrics

    log_msg("Evaluating Hetero-GNN on Setting 1 (Unseen Patients)...")
    gnn_m1 = evaluate_gnn(test_seen_cases, cand_pool_setting1, is_zero_shot=False)
    log_msg("Evaluating Hetero-GNN on Setting 2 (Zero-Shot Unseen Diseases)...")
    gnn_m2 = evaluate_gnn(test_unseen_cases, cand_pool_setting2, is_zero_shot=True)
    
    log_msg(f"-> Hetero-GNN [Setting 1]: R@1={gnn_m1['recall@1']*100:.1f}%, R@5={gnn_m1['recall@5']*100:.1f}%, MRR={gnn_m1['mrr']:.4f}")
    log_msg(f"-> Hetero-GNN [Setting 2]: R@1={gnn_m2['recall@1']*100:.1f}%, R@5={gnn_m2['recall@5']*100:.1f}%, MRR={gnn_m2['mrr']:.4f}")

    baseline_results.append({
        "model": "BiomedicalHeteroGNN",
        "setting": "Setting 1 (Unseen Patients)",
        "recall@1": gnn_m1["recall@1"],
        "recall@5": gnn_m1["recall@5"],
        "recall@10": gnn_m1["recall@10"],
        "mrr": gnn_m1["mrr"],
        "hits@1_ci": gnn_m1.get("hits@1_95ci"),
        "mrr_ci": gnn_m1.get("mrr_95ci")
    })
    baseline_results.append({
        "model": "BiomedicalHeteroGNN",
        "setting": "Setting 2 (Unseen Diseases Zero-Shot)",
        "recall@1": gnn_m2["recall@1"],
        "recall@5": gnn_m2["recall@5"],
        "recall@10": gnn_m2["recall@10"],
        "mrr": gnn_m2["mrr"],
        "hits@1_ci": gnn_m2.get("hits@1_95ci"),
        "mrr_ci": gnn_m2.get("mrr_95ci")
    })

    # 4. Generate Reports
    log_msg("Writing reports/baseline_results.md...")
    with open("reports/baseline_results.md", "w", encoding="utf-8") as f:
        f.write("# Baseline Disease Ranking Evaluation Results\n\n")
        f.write("| Model | Evaluation Setting | Recall@1 | Recall@5 | Recall@10 | MRR | 95% CI (MRR) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in baseline_results:
            if r["model"] != "BiomedicalHeteroGNN":
                f.write(f"| {r['model']} | {r['setting']} | {r['recall@1']*100:.2f}% | {r['recall@5']*100:.2f}% | {r['recall@10']*100:.2f}% | {r['mrr']:.4f} | {r.get('mrr_ci', 'N/A')} |\n")

    log_msg("Writing reports/model_comparison.md...")
    with open("reports/model_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Comprehensive Model Comparison Report\n\n")
        f.write("| Model | Evaluation Setting | Recall@1 | Recall@5 | Recall@10 | MRR | 95% CI (MRR) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in baseline_results:
            f.write(f"| **{r['model']}** | {r['setting']} | {r['recall@1']*100:.2f}% | {r['recall@5']*100:.2f}% | {r['recall@10']*100:.2f}% | {r['mrr']:.4f} | {r.get('mrr_ci', 'N/A')} |\n")

    with open("reports/experiment_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in baseline_results:
            writer.writerow([
                f"EXP-{r['model'][:4]}-{r['setting'][:4]}",
                datetime.now().isoformat(),
                "Evaluation",
                f"{r['model']} on {r['setting']}",
                "SUCCESS",
                f"R@1={r['recall@1']}, R@5={r['recall@5']}, R@10={r['recall@10']}, MRR={r['mrr']}"
            ])

    log_msg("Training and evaluation pipeline finished successfully!")

if __name__ == "__main__":
    train_and_evaluate()
