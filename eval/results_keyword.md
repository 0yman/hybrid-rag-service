# Evaluation results

- Corpus: **22 documents**, **459 chunks**
- Embeddings: `fastembed:sentence-transformers/all-MiniLM-L6-v2` (384d)
- Questions: **15** (15 answerable, 0 out-of-domain)
- Retrieval depth: k = 3
- Generated: 2026-09-23 19:39 UTC

## Retrieval

| Retriever | Hit rate@3 | Recall@3 | Precision@3 | nDCG@3 | MRR |
|---|---|---|---|---|---|
| dense only | 0.800 | 0.800 | 0.267 | 0.717 | 0.689 |
| BM25 only | 1.000 | 1.000 | 0.333 | 0.975 | 0.967 |
| hybrid (RRF) | 0.867 | 0.867 | 0.289 | 0.793 | 0.767 |

### Recall by question type (best retriever)

| Type | Recall |
|---|---|
| acronyms | 1.000 |
| code | 1.000 |
| code_and_year | 1.000 |
| rare_proper_noun | 1.000 |
| technical_term | 1.000 |

No retrieval misses: every answerable question found a relevant document within the top k.

## Retrieval depth sweep

A single k flatters whichever retriever happens to saturate there.

| k | Retriever | Recall@k | MRR | nDCG@k |
|---|---|---|---|---|
| 1 | dense only | 0.600 | 0.600 | 0.600 |
| 1 | BM25 only | 0.933 | 0.933 | 0.933 |
| 1 | hybrid (RRF) | 0.667 | 0.667 | 0.667 |
| 3 | dense only | 0.800 | 0.689 | 0.717 |
| 3 | BM25 only | 1.000 | 0.967 | 0.975 |
| 3 | hybrid (RRF) | 0.867 | 0.767 | 0.793 |
| 5 | dense only | 0.800 | 0.689 | 0.717 |
| 5 | BM25 only | 1.000 | 0.967 | 0.975 |
| 5 | hybrid (RRF) | 1.000 | 0.822 | 0.868 |

## Retriever complementarity (k = 3)

- Found by dense but missed by BM25: **0**
- Found by BM25 but missed by dense: **3**
- Missed by both: **0**

**BM25-only wins**

- `k01` MARPOL STCW MLC conventions
- `k03` Rhakotis Pharos 1900 BC
- `k07` Amoco Cadiz 1978

