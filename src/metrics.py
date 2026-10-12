"""
Evaluation Metrics for Disease Ranking and Candidate Differential
Computes:
- Hits@K / Recall@K (K=1, 5, 10)
- Mean Reciprocal Rank (MRR)
- Precision@K
- NDCG@K
"""

import math
import numpy as np

def compute_ranking_metrics(predictions, k_list=(1, 5, 10)):
    """
    predictions: list of dicts with:
      - 'true_id': ground truth disease ID
      - 'ranked_candidates': list of predicted disease IDs ordered by score descending
    """
    if not predictions:
        return {}
        
    num_queries = len(predictions)
    hits = {k: 0 for k in k_list}
    precisions = {k: 0.0 for k in k_list}
    ndcgs = {k: 0.0 for k in k_list}
    reciprocal_ranks = []
    
    for item in predictions:
        true_id = item["true_id"]
        ranked = item["ranked_candidates"]
        
        # Rank of true disease (1-indexed)
        if true_id in ranked:
            rank = ranked.index(true_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            rank = float("inf")
            reciprocal_ranks.append(0.0)
            
        for k in k_list:
            top_k = ranked[:k]
            if true_id in top_k:
                hits[k] += 1
                precisions[k] += 1.0 / k
                # Binary NDCG: DCG = 1 / log2(rank + 1), IDCG = 1 / log2(1 + 1) = 1
                ndcgs[k] += 1.0 / math.log2(rank + 1)
            else:
                precisions[k] += 0.0
                ndcgs[k] += 0.0
                
    mrr = float(np.mean(reciprocal_ranks))
    results = {
        "num_queries": num_queries,
        "mrr": round(mrr, 4),
        "median_rr": round(float(np.median(reciprocal_ranks)), 4)
    }
    
    for k in k_list:
        results[f"hits@{k}"] = round(hits[k] / num_queries, 4)
        results[f"recall@{k}"] = round(hits[k] / num_queries, 4)  # For single ground-truth target, Recall@K == Hits@K
        results[f"precision@{k}"] = round(precisions[k] / num_queries, 4)
        results[f"ndcg@{k}"] = round(ndcgs[k] / num_queries, 4)
        
    return results

def compute_confidence_intervals(predictions, k_list=(1, 5, 10), n_bootstraps=200, seed=42):
    """Calculates 95% bootstrap confidence intervals for primary metrics."""
    rng = np.random.RandomState(seed)
    n = len(predictions)
    if n == 0:
        return {}
        
    boot_mrrs = []
    boot_hits1 = []
    boot_hits5 = []
    boot_hits10 = []
    
    for _ in range(n_bootstraps):
        sample = [predictions[idx] for idx in rng.randint(0, n, size=n)]
        metrics = compute_ranking_metrics(sample, k_list=k_list)
        boot_mrrs.append(metrics["mrr"])
        boot_hits1.append(metrics.get("hits@1", 0.0))
        boot_hits5.append(metrics.get("hits@5", 0.0))
        boot_hits10.append(metrics.get("hits@10", 0.0))
        
    return {
        "mrr_95ci": (round(float(np.percentile(boot_mrrs, 2.5)), 4), round(float(np.percentile(boot_mrrs, 97.5)), 4)),
        "hits@1_95ci": (round(float(np.percentile(boot_hits1, 2.5)), 4), round(float(np.percentile(boot_hits1, 97.5)), 4)),
        "hits@5_95ci": (round(float(np.percentile(boot_hits5, 2.5)), 4), round(float(np.percentile(boot_hits5, 97.5)), 4)),
        "hits@10_95ci": (round(float(np.percentile(boot_hits10, 2.5)), 4), round(float(np.percentile(boot_hits10, 97.5)), 4))
    }
