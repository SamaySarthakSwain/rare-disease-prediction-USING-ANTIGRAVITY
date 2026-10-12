# Comprehensive Model Comparison Report

| Model | Evaluation Setting | Recall@1 | Recall@5 | Recall@10 | MRR | 95% CI (MRR) |
|---|---|---:|---:|---:|---:|---:|
| **Direct HPO Overlap** | Setting 1 (Unseen Patients) | 86.00% | 97.00% | 98.00% | 0.9127 | (0.8663, 0.9558) |
| **Direct HPO Overlap** | Setting 2 (Unseen Diseases Zero-Shot) | 89.00% | 97.00% | 99.00% | 0.9333 | (0.8967, 0.965) |
| **Jaccard Phenotype Similarity** | Setting 1 (Unseen Patients) | 74.00% | 94.00% | 95.00% | 0.8262 | (0.7511, 0.8838) |
| **Jaccard Phenotype Similarity** | Setting 2 (Unseen Diseases Zero-Shot) | 86.00% | 97.00% | 100.00% | 0.9125 | (0.8698, 0.9514) |
| **Lin BMA** | Setting 1 (Unseen Patients) | 94.00% | 100.00% | 100.00% | 0.9645 | (0.9345, 0.9876) |
| **Lin BMA** | Setting 2 (Unseen Diseases Zero-Shot) | 95.00% | 100.00% | 100.00% | 0.9733 | (0.9483, 0.9933) |
| **Resnik BMA (Patient-Recall)** | Setting 1 (Unseen Patients) | 94.00% | 100.00% | 100.00% | 0.9667 | (0.9433, 0.99) |
| **Resnik BMA (Patient-Recall)** | Setting 2 (Unseen Diseases Zero-Shot) | 91.00% | 100.00% | 100.00% | 0.9490 | (0.92, 0.98) |
| **Weighted Ontology Model** | Setting 1 (Unseen Patients) | 94.00% | 100.00% | 100.00% | 0.9667 | (0.9433, 0.99) |
| **Weighted Ontology Model** | Setting 2 (Unseen Diseases Zero-Shot) | 91.00% | 100.00% | 100.00% | 0.9490 | (0.92, 0.98) |
| **BiomedicalHeteroGNN** | Setting 1 (Unseen Patients) | 0.00% | 3.00% | 7.00% | 0.0371 | (0.027, 0.0458) |
| **BiomedicalHeteroGNN** | Setting 2 (Unseen Diseases Zero-Shot) | 3.00% | 4.00% | 11.00% | 0.0670 | (0.0396, 0.1047) |
| **BiomedicalDeepNeuralRanker (Ours)** | Setting 1 (Unseen Patients) | **98.00%** | **100.00%** | **100.00%** | **0.9900** | (0.975, 1.0) |
| **BiomedicalDeepNeuralRanker (Ours)** | Setting 2 (Unseen Diseases Zero-Shot) | **98.00%** | **100.00%** | **100.00%** | **0.9883** | (0.97, 1.0) |
