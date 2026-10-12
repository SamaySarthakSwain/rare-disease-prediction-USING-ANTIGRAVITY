# Issues and Resolutions Log

### Issue 1: Gene Identifier Namespace Discrepancy
- **Description:** `genes_to_disease.txt` uses `NCBIGene:1234` while `genes_to_phenotype.txt` uses raw integer strings `1234`.
- **Resolution:** In Stage 2 normalization, normalize all NCBI gene IDs to canonical format `NCBI:<id>`.

### Issue 2: Negated Phenotypes in HPOA
- **Description:** 733 records in `phenotype (1).hpoa` contain `qualifier='NOT'`.
- **Resolution:** Never treat negated phenotypes as positive disease associations. Segregate them into explicit negative annotations or filter them out from positive profile matching.

### Issue 3: Absence of Pre-existing Clinical Patient Cases
- **Description:** The four files are reference ontology and association files (HPOA, HPO, gene-disease, gene-phenotype). No independent clinical patient case dataset was provided in the workspace.
- **Resolution:** Establish realistic patient-simulation and masking evaluation protocols (with varying noise, missing symptoms, and unseen-disease holdouts) while transparently documenting the benchmark methodology without fabricating real clinical patients.
