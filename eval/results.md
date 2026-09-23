# Evaluation results

- Corpus: **22 documents**, **459 chunks**
- Embeddings: `fastembed:sentence-transformers/all-MiniLM-L6-v2` (384d)
- Questions: **35** (31 answerable, 4 out-of-domain)
- Retrieval depth: k = 5
- Generated: 2026-09-23 19:39 UTC

## Retrieval

| Retriever | Hit rate@5 | Recall@5 | Precision@5 | nDCG@5 | MRR |
|---|---|---|---|---|---|
| dense only | 1.000 | 0.984 | 0.206 | 0.948 | 0.946 |
| BM25 only | 0.935 | 0.935 | 0.200 | 0.908 | 0.903 |
| hybrid (RRF) | 1.000 | 1.000 | 0.213 | 0.955 | 0.944 |

### Recall by question type (best retriever)

| Type | Recall |
|---|---|
| causal | 1.000 |
| comparison | 1.000 |
| conceptual | 1.000 |
| definition | 1.000 |
| factoid | 1.000 |
| numeric | 1.000 |

No retrieval misses: every answerable question found a relevant document within the top k.

## Retrieval depth sweep

A single k flatters whichever retriever happens to saturate there.

| k | Retriever | Recall@k | MRR | nDCG@k |
|---|---|---|---|---|
| 1 | dense only | 0.871 | 0.903 | 0.903 |
| 1 | BM25 only | 0.839 | 0.871 | 0.871 |
| 1 | hybrid (RRF) | 0.871 | 0.903 | 0.903 |
| 3 | dense only | 0.952 | 0.935 | 0.931 |
| 3 | BM25 only | 0.919 | 0.903 | 0.899 |
| 3 | hybrid (RRF) | 0.968 | 0.935 | 0.941 |
| 5 | dense only | 0.984 | 0.946 | 0.948 |
| 5 | BM25 only | 0.935 | 0.903 | 0.908 |
| 5 | hybrid (RRF) | 1.000 | 0.944 | 0.955 |
| 10 | dense only | 1.000 | 0.946 | 0.956 |
| 10 | BM25 only | 0.968 | 0.910 | 0.920 |
| 10 | hybrid (RRF) | 1.000 | 0.944 | 0.955 |

## Retriever complementarity (k = 5)

- Found by dense but missed by BM25: **2**
- Found by BM25 but missed by dense: **0**
- Missed by both: **0**

**Dense-only wins**

- `q01` What does TEU stand for in shipping?
- `q28` What is the difference between cargo and freight?


## Answer quality

- LLM backend: `extractive`
- Keyword coverage: **0.790**
- Citation rate: **1.000**
- Abstention accuracy: **1.000**
- p50 latency: 189 ms
