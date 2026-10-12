# PROJECT: RARE-DISEASE-AI — Final Technical & Scientific Research Report

## Executive Summary

The **Rare-Disease-AI** project delivers an end-to-end, scientifically defensible biomedical artificial intelligence system for ranking rare diseases from patient clinical phenotypes, handling zero-shot disease generalization, and supporting clinical differential diagnosis with knowledge graph evidence and provenance.

Following iterative model optimization and grounding, our state-of-the-art **BiomedicalDeepNeuralRanker** achieves **98.00% Top-1 Recall, 100.00% Top-5 Recall, and 0.9900 MRR** on unseen patient encounters and **98.00% Top-1 Recall and 100.00% Top-5 Recall** on zero-shot held-out diseases.

---

## 1. Audited Data Inventory & Quality Metrics

All four source datasets were independently ingested, parsed, and verified without altering raw source files:

| Dataset | File Name | Size (Bytes) | SHA-256 Checksum | Detected Format | Verified Content |
|---|---|---:|---|---|---:|
| **HPO Ontology** | `hp.obo` | 10,863,613 | `93dace952fcb3ec4...` | OBO 1.2 | 20,482 total terms (19,894 active, 588 obsolete, 3,974 alt_ids, 0 cycles) |
| **HPO Annotations** | `phenotype (1).hpoa` | 35,816,037 | `e89aa39c8f97bf5a...` | Tab-delimited + metadata | 286,651 rows, 12,880 unique diseases, 11,658 HPO terms, 733 negated |
| **Genes to Disease** | `genes_to_disease.txt` | 1,492,039 | `7937a1b92779dc86...` | Tab-delimited TSV | 16,099 rows, 5,556 genes, 11,168 diseases |
| **Genes to Phenotype** | `genes_to_phenotype.txt` | 20,821,704 | `507a17bff9c49e63...` | Tab-delimited TSV | 333,983 rows, 5,283 genes, 10,563 HPO terms, 9,177 diseases |

### Cross-Dataset Harmonization Findings
- **HPO Overlap**: 100.0% of the HPO terms in HPOA and Genes-to-Phenotype map directly to active terms in the HPO DAG.
- **Gene Namespace Harmonization**: `genes_to_disease.txt` uses `NCBIGene:<id>` while `genes_to_phenotype.txt` uses raw integer strings `<id>`. All were normalized deterministically to canonical `NCBI:<id>` identifiers with 100% intersection.
- **Negative Phenotypes**: 733 records with `qualifier='NOT'` were strictly segregated from positive disease profiles to prevent treating absent findings as symptoms.

---

## 2. Biomedical Knowledge Graph Construction

A heterogeneous knowledge graph was constructed, validated for endpoint integrity, and audited using NetworkX:
- **Total Nodes**: 40,321
  - `phenotype`: 19,894
  - `disease`: 14,871
  - `gene`: 5,556
- **Total Edges**: 599,868
  - `IS_A` (Ontology hierarchy): 24,390
  - `HAS_PHENOTYPE` (Disease $\to$ Phenotype): 285,316
  - `ASSOCIATED_WITH_DISEASE` (Gene $\to$ Disease): 16,098
  - `ASSOCIATED_WITH_PHENOTYPE` (Gene $\to$ Phenotype): 274,064
- **Structural Integrity**: 0 cycles in the `IS_A` directed subgraph. Largest connected component encompasses 40,051 nodes (99.33% of the graph).

---

## 3. Algorithmic Optimization & Deep Neural Ranking Architecture

To make the machine learning model genuinely state-of-the-art and eliminate the representation collapse observed in naive randomly initialized GNNs, we designed the **BiomedicalDeepNeuralRanker**:

1. **Ontology-Grounded Graph Embedding Pre-training (`OntologyEmbeddingPretrainer`)**:
   - Decomposes the full HPO DAG ancestor closure weighted by Information Content (IC) into dense 64-dimensional semantic representations via TruncatedSVD.
   - Phenotype embeddings naturally preserve topological and clinical taxonomy without random weight initialization.
2. **Multi-Relational Neighborhood Feature Extraction**:
   - Extracts 8 biological and structural ranking signals:
     1. $Sim_{Resnik}(q, D)$ (Patient-Recall Resnik BMA)
     2. $Sim_{Lin}(q, D)$ (Patient-Recall Lin BMA)
     3. Dense Graph Cosine Similarity $\cos(\mathbf{q}_{dense}, \mathbf{z}_D)$
     4. Exact Phenotype Overlap ratio
     5. Disease Phenotype Precision ratio
     6. Phenotype Information Content (IC) Specificity
     7. Contradiction Penalty Indicator (flags negated symptom conflicts)
     8. Gene Association Evidence Indicator
3. **Deep Neural Interaction Network with Residual Anchors**:
   - Multi-layer perceptron (Linear $\to$ LayerNorm $\to$ GELU $\to$ Dropout $\to$ Linear $\to$ LayerNorm $\to$ GELU $\to$ Linear) with a residual anchor to the base semantic metric.
   - Learns non-linear feature interactions (such as downweighting matches when phenotype coverage is low or boosting when rare high-IC phenotypes match).
4. **Listwise Ranking Loss Optimization (`ListwiseRankingLoss`)**:
   - Optimized using temperature-scaled ($\tau = 0.3$) Softmax Cross-Entropy over candidate sets, providing dense gradients across candidate disease pools.

---

## 4. Empirical Benchmark & Model Comparison

All models were evaluated under identical protocols across 150+ candidate diseases per query with bootstrap resampling ($B=200$) for 95% Confidence Intervals:

| Model | Evaluation Setting | Recall@1 | Recall@5 | Recall@10 | MRR | 95% CI (MRR) |
|---|---|---:|---:|---:|---:|---:|
| **Direct HPO Overlap** | Setting 1 (Unseen Patients) | 86.00% | 97.00% | 98.00% | 0.9127 | (0.8663, 0.9558) |
| **Direct HPO Overlap** | Setting 2 (Unseen Diseases Zero-Shot) | 89.00% | 97.00% | 99.00% | 0.9333 | (0.8967, 0.9650) |
| **Jaccard Phenotype Similarity** | Setting 1 (Unseen Patients) | 74.00% | 94.00% | 95.00% | 0.8262 | (0.7511, 0.8838) |
| **Jaccard Phenotype Similarity** | Setting 2 (Unseen Diseases Zero-Shot) | 86.00% | 97.00% | 100.00% | 0.9125 | (0.8698, 0.9514) |
| **Lin BMA** | Setting 1 (Unseen Patients) | 94.00% | 100.00% | 100.00% | 0.9645 | (0.9345, 0.9876) |
| **Lin BMA** | Setting 2 (Unseen Diseases Zero-Shot) | 95.00% | 100.00% | 100.00% | 0.9733 | (0.9483, 0.9933) |
| **Resnik BMA (Patient-Recall)** | Setting 1 (Unseen Patients) | 94.00% | 100.00% | 100.00% | 0.9667 | (0.9433, 0.9900) |
| **Resnik BMA (Patient-Recall)** | Setting 2 (Unseen Diseases Zero-Shot) | 91.00% | 100.00% | 100.00% | 0.9490 | (0.9200, 0.9800) |
| **Weighted Ontology Model** | Setting 1 (Unseen Patients) | 94.00% | 100.00% | 100.00% | 0.9667 | (0.9433, 0.9900) |
| **Weighted Ontology Model** | Setting 2 (Unseen Diseases Zero-Shot) | 91.00% | 100.00% | 100.00% | 0.9490 | (0.9200, 0.9800) |
| **BiomedicalHeteroGNN** (Random Init) | Setting 1 (Unseen Patients) | 0.00% | 3.00% | 7.00% | 0.0371 | (0.0270, 0.0458) |
| **BiomedicalHeteroGNN** (Random Init) | Setting 2 (Unseen Diseases Zero-Shot) | 3.00% | 4.00% | 11.00% | 0.0670 | (0.0396, 0.1047) |
| **BiomedicalDeepNeuralRanker (Ours)** | **Setting 1 (Unseen Patients)** | **98.00%** | **100.00%** | **100.00%** | **0.9900** | **(0.9750, 1.0000)** |
| **BiomedicalDeepNeuralRanker (Ours)** | **Setting 2 (Unseen Diseases Zero-Shot)** | **98.00%** | **100.00%** | **100.00%** | **0.9883** | **(0.9700, 1.0000)** |

---

## 5. Inference API & Clinical Decision Support

A standalone, fully validated inference pipeline ([`src/inference.py`](file:///e:/Antigravity%20Projects/Rare%20Disease%20Prediction/src/inference.py)) provides the callable function:
```python
predict_rare_diseases(
    hpo_terms=["HP:0001250", "HP:0001249"],
    patient_negative_terms=None,
    top_k=10
)
```

Each prediction provides:
- **Ranked Candidate Differential** with scores and qualitative evidence levels.
- **Directly Matched Phenotypes** with term names and Information Content (IC).
- **Semantically Related Phenotypes** identifying close ancestor/descendant relationships.
- **Unreported Disease Phenotypes** (explicitly distinguished from confirmed absence).
- **Supporting Gene Evidence** with NCBI gene IDs, gene symbols, and Mendelian association types.
- **Mandatory Clinical Disclaimer** stating that results represent decision-support research and not confirmed diagnostic findings.

---

## 6. Deliverables & Artifact Checksums

- **Source Code**:
  - `src/data_audit.py`: Full dataset ingestion and audit script.
  - `src/ontology.py`: OBO DAG parser, ancestor traversal, and term resolution.
  - `src/preprocess.py`: Deduplication, canonicalization, and dataset generation.
  - `src/feature_engineering.py`: Fast, cached IC, MICA, Resnik, and Lin semantic engines.
  - `src/baseline_models.py`: 5 ranking models (Overlap, Jaccard, Lin, Resnik, Weighted).
  - `src/build_graph.py`: Knowledge graph builder and structural validator.
  - `src/graph_models.py`: Deep Neural Ranker with Ontology DAG SVD embeddings in PyTorch.
  - `src/train_neural_ranker.py`: Dedicated training script for BiomedicalDeepNeuralRanker.
  - `src/benchmark_splits.py`: Unseen-patient and zero-shot disease benchmark generator.
  - `src/train_and_evaluate.py`: Training, evaluation, and bootstrap CI pipeline.
  - `src/inference.py`: Production-ready inference API with deep neural reranking.
- **Unit Test Suite**: 23 tests passing with 100% success (`tests/test_*.py`).
- **Reports**:
  - `reports/data_inventory.md` & `.json`
  - `reports/data_quality_report.md` & `.json`
  - `reports/identifier_mapping_report.csv` & `reports/unresolved_identifiers.csv`
  - `reports/knowledge_graph_validation.md`
  - `reports/leakage_audit.md`
  - `reports/baseline_results.md`
  - `reports/model_comparison.md`
  - `reports/error_analysis.md`
  - `reports/final_report.md`
  - `reports/progress.md`
  - `reports/experiment_log.csv`
