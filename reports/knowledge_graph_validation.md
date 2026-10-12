# Biomedical Knowledge Graph Validation Report

- **Generation Timestamp:** 2026-10-12 20:40:14
- **Total Nodes:** 40,321
- **Total Edges:** 599,868

## Node Distributions

| Node Type | Count |
|---|---:|
| `phenotype` | 19,894 |
| `disease` | 14,871 |
| `gene` | 5,556 |

## Edge Distributions

| Relation Type | Count |
|---|---:|
| `IS_A` | 24,390 |
| `HAS_PHENOTYPE` | 285,316 |
| `ASSOCIATED_WITH_DISEASE` | 16,098 |
| `ASSOCIATED_WITH_PHENOTYPE` | 274,064 |

## Structural Integrity Checks

- **Ontology Hierarchy (`IS_A`) Cycles:** 0 (Strict DAG verified)
- **Connected Components:** 120
- **Largest Connected Component:** 40,051 nodes (99.33% of graph)
- **Isolated Nodes:** 24 (Nodes without verified positive edges in current dataset)

## Graph Validation Status

PASS: All node and edge endpoints exist in node index. No dangling references or ontology cycles detected.
