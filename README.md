# RARE-DISEASE-AI

> End-to-End Biomedical AI Training, Knowledge Graph Learning, Zero-Shot Disease Prediction, and Rigorous Clinical Differential Ranking.

## Overview
This repository contains a reproducible research pipeline designed to rank rare diseases based on patient clinical phenotypes and symptoms, including the ability to retrieve diseases excluded from training (Zero-Shot Generalization).

---

## Repository Structure
```
Rare Disease Prediction/
├── Dataset 1/                    # Human Phenotype Ontology (hp.obo)
├── Dataset 2/                    # HPO Disease Annotations (phenotype (1).hpoa)
├── Dataset 3/                    # Genes to Disease (genes_to_disease.txt)
├── Dataset 4/                    # Genes to Phenotype (genes_to_phenotype.txt)
├── configs/                      # Configuration files
├── src/
│   ├── data_audit.py             # Dataset audit & SHA-256 validation
│   ├── ontology.py               # HPO OBO parser & hierarchy traversal
│   ├── preprocess.py             # Normalization, cleaning & manifest creation
│   ├── build_graph.py            # Heterogeneous Knowledge Graph builder
│   ├── feature_engineering.py    # IC, MICA, Resnik & Lin semantic similarities
│   ├── baseline_models.py        # 5 baseline disease ranking models
│   ├── graph_models.py           # PyTorch Heterogeneous GNN ranking model
│   ├── benchmark_splits.py       # Unseen-patient & zero-shot disease benchmark
│   ├── train_and_evaluate.py     # Training, evaluation & bootstrap CIs
│   ├── metrics.py                # Hits@K, Recall@K, MRR, NDCG@K, CIs
│   └── inference.py              # Callable diagnostic differential inference API
├── tests/                        # 23 unit tests (100% passing)
├── processed/                    # Clean intermediate datasets & graph CSVs
├── models/                       # Saved model checkpoints & configurations
├── reports/                      # Full suite of markdown & JSON audit reports
└── README.md
```

---

## Quickstart & Replication

### 1. Run Data Audit & Quality Inspection
```bash
python src/data_audit.py
```

### 2. Preprocess & Clean Datasets
```bash
python src/preprocess.py
```

### 3. Build Biomedical Knowledge Graph
```bash
python src/build_graph.py
```

### 4. Generate Benchmark Splits & Leakage Audit
```bash
python src/benchmark_splits.py
```

### 5. Train & Evaluate Models
```bash
python src/train_and_evaluate.py
```

### 6. Run Unit Test Suite
```bash
python -m unittest discover tests
```

---

## Python Inference API
```python
from src.inference import predict_rare_diseases

# Clinical query: Patient presenting with Seizures and Intellectual Disability
result = predict_rare_diseases(
    hpo_terms=["HP:0001250", "HP:0001249"],
    top_k=5
)

for d in result["differential_diagnosis"]:
    print(f"Rank {d['rank']}: {d['disease_name']} ({d['disease_id']}) - Score: {d['ranking_score']}")
```

---

## Disclaimer
**RESEARCH PROTOTYPE ONLY**: This system is a biomedical research prototype and clinical decision-support experiment, not a clinically validated diagnostic device. Predictions must not be represented as confirmed diagnoses.
