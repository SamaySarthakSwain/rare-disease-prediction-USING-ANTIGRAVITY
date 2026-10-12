# Data Quality & Cross-Dataset Audit Report

Audit completed: 2026-10-12T15:07:09.626400Z

## 1. Human Phenotype Ontology (`hp.obo`)
- **Total terms parsed:** 20,482
- **Active terms:** 19,894
- **Obsolete terms:** 588
- **Alternative identifiers (alt_ids):** 3,974
- **Replaced_by mappings:** 507
- **Cycles in hierarchy:** 0 (Verified Directed Acyclic Graph)
- **Namespaces:** {}

## 2. Phenotype Annotations (`phenotype (1).hpoa`)
- **Total annotation rows:** 286,651
- **Unique diseases:** 12,880
- **Unique HPO terms:** 11,658
- **Explicitly negated phenotypes (qualifier='NOT'):** 733
- **Duplicate (disease, hpo, qual) entries:** 602
- **Evidence code breakdown:** {"TAS": 137011, "PCS": 119333, "IEA": 30307}
- **Aspect breakdown:** {"P": 268906, "C": 8559, "I": 8978, "M": 77, "H": 131}

## 3. Genes to Disease (`genes_to_disease.txt`)
- **Total rows:** 16,099
- **Unique NCBI genes:** 5,556
- **Unique gene symbols:** 5,550
- **Unique diseases:** 11,168
- **Duplicate (gene, disease) pairs:** 1
- **Association types:** {"MENDELIAN": 7094, "POLYGENIC": 582, "UNKNOWN": 8423}

## 4. Genes to Phenotype (`genes_to_phenotype.txt`)
- **Total rows:** 333,983
- **Unique NCBI genes:** 5,283
- **Unique gene symbols:** 5,277
- **Unique HPO terms:** 10,563
- **Associated disease identifiers:** 9,177
- **Duplicate rows:** 0

## 5. Cross-Dataset Identifier Overlap & Harmonization
- **HPOA HPO terms in active ontology:** 11,658 / 11,658 (100.00%)
- **HPOA HPO terms mapped via alt_id:** 0
- **HPOA HPO terms obsolete / unmapped:** 0 / 0
- **G2P HPO terms in active ontology:** 10,563 / 10,563 (100.00%)
- **Disease overlap between HPOA and Genes-to-Disease:** 9,177 diseases shared
- **Diseases in HPOA only (phenotypes but no gene mapping):** 3,703
- **Diseases in G2D only (genes but no HPOA phenotypes):** 1,991
- **NCBI Gene ID format difference:** G2D uses `NCBIGene:<id>` while G2P uses `<id>` (integer string). When normalized, ID overlap = 5,283 genes.
