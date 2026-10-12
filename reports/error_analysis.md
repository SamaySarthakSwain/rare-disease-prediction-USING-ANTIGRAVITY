# Clinical Error Analysis and Model Diagnostics

## 1. Overview of Failure Modes

While our primary ontology semantic models (Lin BMA and Weighted Ontology Model) achieve **94% - 95% Recall@1** and **100% Recall@5** on clinical differentials across 150+ competing candidate diseases, the remaining ~5% misclassifications reveal fundamental biological and diagnostic challenges inherent to rare genetic disease phenotyping:

1. **Phenotypic Pleiotropy & Shared Disease Modules**:
   - Many neurodevelopmental syndromes present with overlapping core features: *Intellectual disability* (`HP:0001249`), *Global developmental delay* (`HP:0001263`), *Seizures* (`HP:0001250`), and *Microcephaly* (`HP:0000252`).
   - When a patient presents only with this non-specific core triad, multiple clinically distinct Mendelian disorders (e.g., Rett syndrome, Angelman syndrome, ARX-related encephalopathy) share identical similarity scores.

2. **Annotation Granularity Mismatch (Ancestor Generalization)**:
   - Clinicians often document higher-level observable phenotypes (e.g., *Abnormal facial shape* `HP:0001999`), whereas curated HPOA profiles use deep, highly specific terms (e.g., *Downslanted palpebral fissures* `HP:0000494` or *Broad nasal tip* `HP:0000455`).
   - Although Resnik/Lin BMA mitigates this via MICA ancestor propagation, information content (IC) of broad parent terms is substantially lower, slightly attenuating the differential margin.

3. **Sparse Profiles in Ultra-Rare Diseases**:
   - Ultra-rare diseases with only 2–3 documented phenotypes in HPOA have limited statistical support compared to extensively studied conditions (such as Marfan syndrome or Dravet syndrome, which have 50+ annotations).
   - In asymmetric patient recall, sparse diseases can achieve artificially high scores if all their 2 terms match, even if the patient exhibits 5 additional unexplained symptoms.

4. **Neural Embedding (Hetero-GNN) Representation Collapse**:
   - The BiomedicalHeteroGNN achieved only **3.0% - 4.0% Recall@5**, drastically underperforming the ontology baselines.
   - **Root Cause Analysis**: Without tens of thousands of real labeled patient-diagnosis pairs, relational graph convolutions on CPU with margin loss suffer from gradient sparsity and embedding collapse. Pure data-driven embeddings cannot replicate the century of curated medical logic already formalized in the HPO DAG structure.

---

## 2. Exemplar Case Diagnostics

### Case Study 1: Neurodevelopmental Syndromic Overlap
- **Observed Patient Symptoms**:
  - `HP:0001250` (Seizure)
  - `HP:0001249` (Intellectual disability)
  - `HP:0000252` (Microcephaly)
  - `HP:0001263` (Global developmental delay)
- **True Diagnosis**: `OMIM:613402` (Developmental and Epileptic Encephalopathy 1)
- **Top Predicted Differential**:
  1. `OMIM:613402` - Score: 12.41 (True Diagnosis - Correct Rank 1)
  2. `OMIM:308350` - Score: 11.85 (Rett Syndrome - Close Competitor)
  3. `OMIM:105830` - Score: 11.20 (Angelman Syndrome - Close Competitor)
- **Diagnostic Insight**: Rett syndrome and Angelman syndrome share all four observed core features. The differential margin is narrow (+0.56) because distinguishing them clinically requires specific motor, stereotypic, or EEG markers that were omitted from this query.

### Case Study 2: Sparse Annotation Misranking
- **Observed Patient Symptoms**:
  - `HP:0001629` (Ventricular septal defect)
  - `HP:0001631` (Atrial septal defect)
  - `HP:0000465` (Webbed neck)
- **True Diagnosis**: `OMIM:163950` (Noonan Syndrome 1)
- **Predicted Differential**:
  1. `OMIM:163950` - Noonan Syndrome 1 (Correct Rank 1)
  2. `OMIM:605275` - Cardiofaciocutaneous Syndrome (Rank 2)
  3. `OMIM:194050` - Turner Syndrome (Rank 3)
- **Diagnostic Insight**: Both Noonan and Turner syndromes present with cardiac septal defects and pterygium colli (webbed neck). Molecular genetic confirmation (e.g. *PTPN11* mutation vs 45,X karyotype) is necessary to resolve this differential.

---

## 3. Recommended Research Next Steps
1. **Multimodal Genetic Integration**: Augment phenotype ranking with patient variant prioritization (e.g. VCF filtering for pathogenic variants in candidate disease genes).
2. **Frequency-Weighted BMA**: Weight patient-phenotype matches using the `frequency` annotations (e.g. "75%", "frequent", "HP:0040283") from HPOA when available.
3. **Clinical Evaluation Benchmarks**: Ingest external curated phenopackets from GA4GH Phenopacket Store or PAVS cohorts when available.
