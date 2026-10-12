# Data Leakage Audit and Benchmark Protocol

## 1. Disease Zero-Shot Partitioning
- **Total Eligible Diseases (>=3 phenotypes):** 12,303
- **Train Diseases:** 9,842 (80.0%)
- **Validation Diseases:** 1,230 (10.0%)
- **Test Diseases (Zero-Shot):** 1,231 (10.0%)
- **Intersection check:** `train & test` = 0, `train & val` = 0, `val & test` = 0 (Strictly disjoint)

## 2. Evaluation Cases
- **Training Cases (Seen diseases):** 500
- **Validation Cases (Seen diseases):** 250
- **Test Cases (Setting 1 - Unseen Patients, Seen Diseases):** 250
- **Test Cases (Setting 2 - Unseen Diseases Zero-Shot):** 500

## 3. Leakage Prevention Protocol
1. **Information Content (IC) Estimation:** IC statistics are fit solely on annotations of `train_diseases`.
2. **Zero-Shot Disease Labeling:** Test diseases are strictly excluded from any supervised disease classifier training.
3. **Graph Feature Separation:** Models evaluated under strict zero-shot conditions cannot access disease identities during training.
4. **Provenance Disclosure:** Cases are explicitly marked as `association_derived_simulation` with masking and noise; results are scientifically reported as benchmark simulations, not unverified clinical diagnostic trials.
