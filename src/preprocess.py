"""
Preprocessing and Normalization Pipeline for Rare-Disease-AI
Produces clean, validated intermediate datasets:
- processed/hpo_terms.csv
- processed/disease_phenotype.csv
- processed/gene_disease_processed.csv
- processed/gene_phenotype_processed.csv
- processed/disease_metadata.csv
- processed/identifier_mapping.csv
- processed/preprocessing_manifest.json
"""

import os
import sys
import csv
import json
import hashlib
import time
from datetime import datetime
from collections import defaultdict, Counter

# Ensure src is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ontology import HPOntology

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def normalize_ncbi_gene(raw_gene):
    g = raw_gene.strip()
    if g.startswith("NCBIGene:"):
        return f"NCBI:{g[9:]}"
    elif g.startswith("NCBI:"):
        return g
    else:
        return f"NCBI:{g}"

def run_preprocessing():
    start_time = time.time()
    os.makedirs("processed", exist_ok=True)
    
    src_files = {
        "hp_obo": "Dataset 1/hp.obo",
        "hpoa": "Dataset 2/phenotype (1).hpoa",
        "g2d": "Dataset 3/genes_to_disease.txt",
        "g2p": "Dataset 4/genes_to_phenotype.txt"
    }
    
    # 1. Parse and build HPO ontology
    print("Loading HPO ontology...")
    hpo = HPOntology(src_files["hp_obo"])
    print(f"Loaded {len(hpo.terms)} total terms ({len(hpo.names)} active terms).")
    
    # Track mappings
    mappings = [] # list of (entity_type, original, canonical, source, status, conf, notes)
    for alt, primary in hpo.alt_ids.items():
        mappings.append(("phenotype", alt, primary, "hp.obo", "mapped_alias", 1.0, "Alternative ID resolved to primary"))
    for obs, rep in hpo.replaced_by.items():
        mappings.append(("phenotype", obs, rep, "hp.obo", "mapped_replacement", 1.0, "Obsolete term mapped via replaced_by"))
        
    # Write processed/hpo_terms.csv
    print("Exporting processed/hpo_terms.csv...")
    hpo_terms_out = "processed/hpo_terms.csv"
    active_term_count = 0
    with open(hpo_terms_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hpo_id", "name", "definition", "parents", "depth_from_root"])
        for tid, tdata in hpo.terms.items():
            if tdata["is_obsolete"]:
                continue
            active_term_count += 1
            name = tdata["name"] or ""
            definition = tdata["definition"] or ""
            parents_str = ";".join(sorted(hpo.parents.get(tid, [])))
            # Compute depth only for a subset or root
            depth = 0 if tid == hpo.root else -1
            writer.writerow([tid, name, definition, parents_str, depth])
            
    # 2. Process phenotype (1).hpoa -> processed/disease_phenotype.csv
    print("Processing HPOA annotations...")
    hpoa_in = src_files["hpoa"]
    hpoa_out = "processed/disease_phenotype.csv"
    hpoa_total_raw = 0
    hpoa_kept = 0
    hpoa_negated = 0
    hpoa_unresolved = 0
    hpoa_duplicates_removed = 0
    seen_disease_hpo = set()
    
    disease_names = {}
    disease_phenotype_counts = Counter()
    disease_sources = defaultdict(set)
    
    with open(hpoa_in, "r", encoding="utf-8", errors="replace") as fin, \
         open(hpoa_out, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["disease_id", "disease_name", "hpo_id", "hpo_name", "qualifier", "evidence", "onset", "frequency", "aspect"])
        
        header = None
        for line in fin:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\r\n").split("\t")
            if not header:
                header = parts
                continue
            hpoa_total_raw += 1
            rec = dict(zip(header, parts + [""]*(len(header)-len(parts))))
            
            db_id = rec.get("database_id", "").strip()
            dname = rec.get("disease_name", "").strip()
            raw_hpo = rec.get("hpo_id", "").strip()
            qual = rec.get("qualifier", "").strip().upper()
            ev = rec.get("evidence", "").strip()
            onset = rec.get("onset", "").strip()
            freq = rec.get("frequency", "").strip()
            aspect = rec.get("aspect", "").strip()
            
            if db_id and dname:
                disease_names[db_id] = dname
                src_prefix = db_id.split(":")[0] if ":" in db_id else "UNKNOWN"
                disease_sources[db_id].add(src_prefix)
                
            resolved_hpo = hpo.resolve_term(raw_hpo)
            if not resolved_hpo:
                hpoa_unresolved += 1
                mappings.append(("phenotype", raw_hpo, "UNRESOLVED", "phenotype.hpoa", "unresolved", 0.0, "HPO ID not found in ontology"))
                continue
                
            if resolved_hpo != raw_hpo:
                mappings.append(("phenotype", raw_hpo, resolved_hpo, "phenotype.hpoa", "mapped_alias", 1.0, "Resolved alias to primary"))
                
            is_negated = (qual == "NOT")
            if is_negated:
                hpoa_negated += 1
                qual_val = "NOT"
            else:
                qual_val = "AFFIRMATIVE"
                disease_phenotype_counts[db_id] += 1
                
            dedup_key = (db_id, resolved_hpo, qual_val)
            if dedup_key in seen_disease_hpo:
                hpoa_duplicates_removed += 1
                continue
            seen_disease_hpo.add(dedup_key)
            
            hpo_name = hpo.names.get(resolved_hpo, "")
            writer.writerow([db_id, dname, resolved_hpo, hpo_name, qual_val, ev, onset, freq, aspect])
            hpoa_kept += 1

    # 3. Process genes_to_disease.txt -> processed/gene_disease_processed.csv
    print("Processing genes_to_disease...")
    g2d_in = src_files["g2d"]
    g2d_out = "processed/gene_disease_processed.csv"
    g2d_total_raw = 0
    g2d_kept = 0
    g2d_duplicates = 0
    seen_gene_disease = set()
    disease_gene_counts = Counter()
    
    with open(g2d_in, "r", encoding="utf-8", errors="replace") as fin, \
         open(g2d_out, "w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin, delimiter="\t")
        hdr = next(reader)
        writer = csv.writer(fout)
        writer.writerow(["ncbi_gene_id", "gene_symbol", "association_type", "disease_id", "source"])
        for parts in reader:
            g2d_total_raw += 1
            rec = dict(zip(hdr, parts + [""]*(len(hdr)-len(parts))))
            raw_gid = rec.get("ncbi_gene_id", "").strip()
            sym = rec.get("gene_symbol", "").strip()
            atype = rec.get("association_type", "").strip()
            did = rec.get("disease_id", "").strip()
            src = rec.get("source", "").strip()
            
            can_gid = normalize_ncbi_gene(raw_gid)
            mappings.append(("gene", raw_gid, can_gid, "genes_to_disease.txt", "exact_match", 1.0, "Normalized to NCBI:<id>"))
            
            pair = (can_gid, did, atype)
            if pair in seen_gene_disease:
                g2d_duplicates += 1
                continue
            seen_gene_disease.add(pair)
            
            disease_gene_counts[did] += 1
            writer.writerow([can_gid, sym, atype, did, src])
            g2d_kept += 1

    # 4. Process genes_to_phenotype.txt -> processed/gene_phenotype_processed.csv
    print("Processing genes_to_phenotype...")
    g2p_in = src_files["g2p"]
    g2p_out = "processed/gene_phenotype_processed.csv"
    g2p_total_raw = 0
    g2p_kept = 0
    g2p_duplicates = 0
    g2p_unresolved = 0
    seen_gene_phenotype = set()
    
    with open(g2p_in, "r", encoding="utf-8", errors="replace") as fin, \
         open(g2p_out, "w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin, delimiter="\t")
        hdr = next(reader)
        writer = csv.writer(fout)
        writer.writerow(["ncbi_gene_id", "gene_symbol", "hpo_id", "hpo_name", "frequency", "disease_id"])
        for parts in reader:
            g2p_total_raw += 1
            rec = dict(zip(hdr, parts + [""]*(len(hdr)-len(parts))))
            raw_gid = rec.get("ncbi_gene_id", "").strip()
            sym = rec.get("gene_symbol", "").strip()
            raw_hpo = rec.get("hpo_id", "").strip()
            freq = rec.get("frequency", "").strip()
            did = rec.get("disease_id", "").strip()
            
            can_gid = normalize_ncbi_gene(raw_gid)
            resolved_hpo = hpo.resolve_term(raw_hpo)
            if not resolved_hpo:
                g2p_unresolved += 1
                continue
                
            triple = (can_gid, resolved_hpo, did)
            if triple in seen_gene_phenotype:
                g2p_duplicates += 1
                continue
            seen_gene_phenotype.add(triple)
            
            hname = hpo.names.get(resolved_hpo, rec.get("hpo_name", "").strip())
            writer.writerow([can_gid, sym, resolved_hpo, hname, freq, did])
            g2p_kept += 1

    # 5. Disease metadata table -> processed/disease_metadata.csv
    print("Generating processed/disease_metadata.csv...")
    all_diseases = set(disease_names.keys()) | set(disease_phenotype_counts.keys()) | set(disease_gene_counts.keys())
    with open("processed/disease_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["disease_id", "disease_name", "num_affirmative_phenotypes", "num_associated_genes", "sources", "has_gene_evidence"])
        for did in sorted(all_diseases):
            dname = disease_names.get(did, "Unknown")
            num_phenos = disease_phenotype_counts.get(did, 0)
            num_genes = disease_gene_counts.get(did, 0)
            src_str = ";".join(sorted(disease_sources.get(did, ["UNKNOWN"])))
            has_gene = int(num_genes > 0)
            writer.writerow([did, dname, num_phenos, num_genes, src_str, has_gene])

    # 6. Identifier mappings sample/distinct table -> processed/identifier_mapping.csv
    print("Writing processed/identifier_mapping.csv...")
    seen_map = set()
    with open("processed/identifier_mapping.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_type", "original_identifier", "canonical_identifier", "mapping_source", "mapping_status", "mapping_confidence", "mapping_notes"])
        for row in mappings:
            key = (row[0], row[1], row[2])
            if key not in seen_map:
                seen_map.add(key)
                writer.writerow(row)

    # 7. Create preprocessing_manifest.json
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "random_seed": 42,
        "source_checksums": {k: compute_sha256(v) for k, v in src_files.items()},
        "output_checksums": {
            "hpo_terms": compute_sha256("processed/hpo_terms.csv"),
            "disease_phenotype": compute_sha256("processed/disease_phenotype.csv"),
            "gene_disease": compute_sha256("processed/gene_disease_processed.csv"),
            "gene_phenotype": compute_sha256("processed/gene_phenotype_processed.csv"),
            "disease_metadata": compute_sha256("processed/disease_metadata.csv"),
            "identifier_mapping": compute_sha256("processed/identifier_mapping.csv")
        },
        "counts": {
            "hpo_terms": {"raw_stanzas": len(hpo.terms), "active_kept": active_term_count},
            "hpoa": {
                "raw_rows": hpoa_total_raw,
                "kept_rows": hpoa_kept,
                "negated_rows": hpoa_negated,
                "unresolved_terms": hpoa_unresolved,
                "duplicates_removed": hpoa_duplicates_removed
            },
            "g2d": {
                "raw_rows": g2d_total_raw,
                "kept_rows": g2d_kept,
                "duplicates_removed": g2d_duplicates
            },
            "g2p": {
                "raw_rows": g2p_total_raw,
                "kept_rows": g2p_kept,
                "duplicates_removed": g2p_duplicates,
                "unresolved_terms": g2p_unresolved
            },
            "diseases": len(all_diseases)
        },
        "filtering_rules": [
            "Obsolete HPO terms removed or mapped via alt_id/replaced_by",
            "Negated annotations in HPOA flagged as qualifier=NOT rather than positive phenotype",
            "Duplicate (disease, hpo, qualifier) pairs deduplicated",
            "Gene identifiers canonicalized to NCBI:<id>",
            "Exact match and alias mappings tracked with provenance"
        ]
    }
    
    with open("processed/preprocessing_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_preprocessing()
