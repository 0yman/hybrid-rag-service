# Hybrid RAG Service

Retrieval-augmented question answering over a document corpus, with **hybrid
retrieval** (dense + BM25 fused with Reciprocal Rank Fusion), **mandatory
citations**, **abstention** on out-of-corpus questions, and an **evaluation
harness that measures whether any of it actually works**.

The retrieval code is the ordinary part. The evaluation is the point: it is
what turned "hybrid retrieval is better" from an assumption into a measured,
and partly *falsified*, claim — see [Results](#results).

```
                    ┌─────────────┐
  question ────────▶│  embed(q)   │──▶ FAISS IndexFlatIP ──┐
                    └─────────────┘      (cosine, 384d)     │
                                                            ├──▶ RRF ──▶ top-k ──▶ LLM ──▶ answer + [citations]
                    ┌─────────────┐                         │                              └─ or abstention
  question ────────▶│  tokenize   │──▶ BM25Okapi ───────────┘
                    └─────────────┘
```

---

## Quickstart

No API key needed — the defaults run the whole pipeline offline.

```bash
pip install -r requirements-dev.txt

python scripts/fetch_corpus.py     # 22 Wikipedia articles on shipping & ports
python scripts/ingest.py           # 459 chunks, ~19s on CPU
python -m pytest                   # 87 tests, ~0.5s, no network

uvicorn rag.api:app --app-dir src --port 8000
```

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "What is demurrage in vessel chartering?"}' | jq
```

For real generated answers, get a [free Gemini API key](https://aistudio.google.com/apikey)
(no credit card) and set `GOOGLE_API_KEY` plus `RAG_LLM_BACKEND=gemini` in `.env`.

Or just `docker compose up --build` — the image builds its own index, so the
container starts ready to serve with no key at all.

---

## Results

Two query sets, because **users ask questions in more than one shape** and a
retriever that is excellent at one can be useless at the other:

| Query set | What it looks like | Size |
|---|---|---|
| `eval/golden.jsonl` | natural-language questions — *"What is the inverse of demurrage called?"* | 31 answerable + 4 out-of-domain |
| `eval/golden_keyword.jsonl` | terse keyword queries — *"MARPOL STCW MLC conventions"* | 15 |

### Retrieval, at k = 3

| Retriever | Natural language | Keyword | **Worst case** |
|---|---|---|---|
| Dense only (MiniLM-L6) | 0.935 | 0.667 | **0.667** |
| BM25 only | 0.919 | **1.000** | **0.919** |
| Hybrid (RRF) | **0.968** | 0.867 | **0.867** |

*(recall@3; full tables including MRR and nDCG in [`eval/results.md`](eval/results.md)
and [`eval/results_keyword.md`](eval/results_keyword.md))*

### What this actually shows

**Dense retrieval is the riskiest single choice.** It wins narrowly on
natural-language questions and then collapses to 0.667 on keyword queries —
worse at k=5 (0.733) than BM25 is at k=1 (0.933). Its failures are exactly the
ones you would predict: rare proper nouns and acronym strings that appear a
handful of times in the corpus.

Five queries BM25 retrieved and dense missed entirely at k=3:

```
k01  MARPOL STCW MLC conventions
k02  twistlocks 3 inches gap 40-foot
k03  Rhakotis Pharos 1900 BC
k07  Amoco Cadiz 1978
k15  slotted facility loading docks
```

**Fusion did not straightforwardly win, and the harness is what revealed it.**
RRF is second-best on both sets rather than best on either — it inherits some
of dense's blind spot and gives up some of BM25's precision. On
natural-language questions at k=5 the complementarity analysis found dense
retrieving *everything* BM25 found plus two more, meaning BM25 contributed
nothing unique and fusing was pure overhead at that depth.

**On this corpus, BM25 alone has the best worst case.** That is not the result
I expected when I built the pipeline, and it is the honest reading of the
numbers. Wikipedia articles are topically distinct and share heavy vocabulary
with questions about them, which is close to the best case for a lexical
retriever. On a corpus with more paraphrase and more synonymy — support
tickets, policy documents, transcripts — the balance would be expected to move
back toward dense, which is exactly why the harness takes a `--golden` flag
rather than hard-coding one query set.

### Answer quality

| Metric | Value | What it measures |
|---|---|---|
| Abstention accuracy | 0.943 | Says "I don't know" on out-of-domain questions instead of inventing an answer |
| Citation rate | 0.968 | Answers carrying at least one valid `[n]` marker |
| Keyword coverage | 0.726 | Expected facts actually present in the answer |

Measured with the deterministic `mock` backend so the numbers are reproducible
in CI. Swap in `RAG_LLM_BACKEND=gemini` for fluent answers; the abstention and
citation contracts are enforced in code either way.

### Reproduce

```bash
make eval-all      # writes eval/results.md and eval/results_keyword.md
```

---

## Design decisions

**Why RRF instead of weighted score blending.** Dense cosine scores live in
`[-1, 1]`; BM25 scores are unbounded and shift with corpus statistics. Adding
them means normalising two incomparable scales and re-tuning that
normalisation whenever the corpus changes. RRF discards scores and keeps only
ranks — `score(d) = Σ weight / (k + rank(d))` — so a document ranked well by
both beats one ranked first by only one. A test asserts that multiplying a
BM25 score by 1000 leaves the fused ordering unchanged.

**Why abstention is a first-class output, not a prompt suggestion.** A RAG
system that always answers is one that hallucinates confidently on anything
outside its corpus. The model is instructed to emit a sentinel token, the
pipeline turns that into `abstained=True`, and the out-of-domain questions in
the golden set exist solely to score it.

**Why citation markers are positional and validated.** The model sees `[1]…[5]`,
never internal chunk ids. Markers outside that range are stripped from the
answer rather than passed through — an invented `[9]` looks checkable to a
reader while pointing at nothing.

**Why sentence-aware chunking.** Fixed character windows cut sentences, and
therefore facts, in half. Chunks pack whole sentences to a word budget and
carry a sentence tail into the next chunk, so a fact spanning a boundary
survives in both. Two bugs in this were caught by tests, not by reading the
code: a closing quotation mark being silently dropped, and small-tail merging
that could exceed the word budget.

**Why `IndexFlatIP` rather than IVF/HNSW.** Flat is exact. At 459 chunks an
approximate index trades recall for a speed-up that is not measurable. The
swap is one line when the corpus justifies it.

**Why three embedding backends.** `local` (sentence-transformers) for real
offline use, `gemini` for the hosted free tier, and `hash` — a deterministic
hashing vectoriser — so the test suite needs no model download, no network and
no API key. CI runs in seconds and fails only for real reasons.

---

## Layout

```
src/rag/
  config.py       Pydantic settings, one place for every knob
  loaders.py      .txt / .md / .pdf  ->  Document
  chunking.py     sentence-aware chunking with overlap
  embeddings.py   gemini | local | hash, all L2-normalised
  vectorstore.py  FAISS IndexFlatIP + persistence
  lexical.py      BM25Okapi index
  fusion.py       Reciprocal Rank Fusion
  retriever.py    hybrid retrieval, optional cross-encoder rerank
  generator.py    grounded prompt, citation parsing, abstention
  pipeline.py     ingest -> index -> retrieve -> generate
  api.py          FastAPI: /query /ingest /health /stats /metrics

eval/
  metrics.py      recall@k, precision@k, MRR, nDCG@k  (pure functions)
  golden.jsonl            natural-language question set
  golden_keyword.jsonl    keyword / exact-token query set
  run_eval.py     ablations, k-sweep, complementarity analysis
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /query` | Question in; answer, citations, and every retrieved context with its per-retriever scores out |
| `POST /ingest` | Index a file or directory at runtime |
| `GET /health` | Liveness + index stats; succeeds even with no index loaded |
| `GET /stats` | Corpus size, embedder, dimensions |
| `GET /metrics` | Prometheus |

`/query` returns the component scores behind every result (`dense`,
`lexical`, and each retriever's rank), which is what makes a bad ranking
diagnosable after the fact rather than a mystery.

## Testing

87 tests, no network, no API key, ~0.5s.

```bash
python -m pytest
python -m ruff check src eval scripts tests
```

CI runs the suite on Python 3.11 and 3.12, rebuilds the index from the
committed corpus, reruns both evaluation suites, and **fails the build if
recall drops below a floor** — so a change that quietly degrades retrieval
shows up in the pull request.

## Limitations

- **Small sample.** 46 queries over 22 documents. Differences of a few points
  between retrievers are within noise; the dense-vs-BM25 gap on keyword
  queries (0.667 vs 1.000) is not.
- **Binary relevance, document-level.** A chunk either comes from a relevant
  document or it does not. Graded relevance would be a better signal for nDCG.
- **The golden sets are author-written**, which risks encoding the same
  assumptions the retriever was built on. The keyword set was added
  specifically because the first set turned out to flatter dense retrieval.
- **No reranker in the reported numbers.** `CrossEncoderReranker` is wired in
  and selectable, but adds a model download, so the committed results are
  without it.
- **Corpus is Wikipedia**, CC BY-SA 4.0, attributed per file.

## License

MIT
