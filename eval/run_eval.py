"""Evaluate retrieval and answer quality against eval/golden.jsonl.

Two layers, because they fail for different reasons and a single number hides
which one broke:

* **Retrieval** - recall@k, MRR, nDCG@k, hit rate. Measured against the source
  documents a human marked relevant. If recall is low, no prompt will save it.
* **Answering** - abstention accuracy, keyword coverage and citation validity.
  Abstention is scored on deliberately out-of-domain questions: a system that
  answers those is hallucinating, however good its retrieval looks.

The `--ablate` mode reruns retrieval with dense only, BM25 only and both
fused, which is the only honest way to claim hybrid retrieval was worth it.

    python eval/run_eval.py --ablate
    python eval/run_eval.py --generate --llm-backend mock
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from metrics import (  # noqa: E402
    hit_rate_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.config import get_settings  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

GOLDEN_PATH = REPO_ROOT / "eval" / "golden.jsonl"

ABLATIONS: dict[str, tuple[str, ...]] = {
    "dense only": ("dense",),
    "BM25 only": ("lexical",),
    "hybrid (RRF)": ("dense", "lexical"),
}


def load_golden(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No evaluation records in {path}")
    return records


def evaluate_retrieval(
    pipeline: RAGPipeline,
    golden: list[dict[str, Any]],
    k: int,
    sources: tuple[str, ...],
) -> dict[str, Any]:
    """Retrieval metrics over the answerable questions only.

    Out-of-domain questions have no relevant document by construction, so
    including them would just divide every score by a constant.
    """
    per_query: list[dict[str, float]] = []
    by_type: dict[str, list[float]] = defaultdict(list)
    failures: list[dict[str, Any]] = []

    for record in golden:
        if not record["answerable"]:
            continue
        contexts, _ = pipeline.retriever.retrieve(record["question"], k, sources)
        retrieved = [c.chunk.source for c in contexts]
        # A source can appear in several retrieved chunks; rank metrics need
        # each document counted once, at its best position.
        deduped: list[str] = []
        for source in retrieved:
            if source not in deduped:
                deduped.append(source)

        relevant = record["relevant_sources"]
        scores = {
            f"hit_rate@{k}": hit_rate_at_k(deduped, relevant, k),
            f"recall@{k}": recall_at_k(deduped, relevant, k),
            f"precision@{k}": precision_at_k(deduped, relevant, k),
            f"ndcg@{k}": ndcg_at_k(deduped, relevant, k),
            "mrr": reciprocal_rank(deduped, relevant),
        }
        per_query.append(scores)
        by_type[record["type"]].append(scores[f"recall@{k}"])

        if scores[f"hit_rate@{k}"] == 0.0:
            failures.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "expected": relevant,
                    "retrieved": deduped[:k],
                }
            )

    keys = [f"hit_rate@{k}", f"recall@{k}", f"precision@{k}", f"ndcg@{k}", "mrr"]
    return {
        "aggregate": {key: mean(q[key] for q in per_query) for key in keys},
        "recall_by_type": {t: mean(v) for t, v in sorted(by_type.items())},
        "failures": failures,
        "queries": len(per_query),
    }


def evaluate_answers(
    pipeline: RAGPipeline, golden: list[dict[str, Any]], k: int
) -> dict[str, Any]:
    """Answer-level metrics, including whether the system knows to abstain."""
    answerable_hits: list[float] = []
    citation_ok: list[float] = []
    abstain_correct: list[float] = []
    latencies: list[float] = []
    rows: list[dict[str, Any]] = []

    for record in golden:
        answer = pipeline.query(record["question"], k)
        latencies.append(answer.latency_ms)

        if record["answerable"]:
            # Did the answer actually contain the facts we asked for?
            text = answer.text.lower()
            keywords = record["answer_keywords"]
            covered = (
                sum(1 for kw in keywords if kw.lower() in text) / len(keywords)
                if keywords else 0.0
            )
            answerable_hits.append(covered)
            citation_ok.append(1.0 if answer.cited_ordinals else 0.0)
            abstain_correct.append(0.0 if answer.abstained else 1.0)
        else:
            # The only correct behaviour on an out-of-domain question.
            abstain_correct.append(1.0 if answer.abstained else 0.0)

        rows.append(
            {
                "id": record["id"],
                "question": record["question"],
                "answerable": record["answerable"],
                "abstained": answer.abstained,
                "citations": answer.cited_ordinals,
                "answer": answer.text[:400],
            }
        )

    return {
        "keyword_coverage": mean(answerable_hits),
        "citation_rate": mean(citation_ok),
        "abstention_accuracy": mean(abstain_correct),
        "p50_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
        "rows": rows,
    }


def retrieved_sources(
    pipeline: RAGPipeline, question: str, k: int, sources: tuple[str, ...]
) -> list[str]:
    contexts, _ = pipeline.retriever.retrieve(question, k, sources)
    deduped: list[str] = []
    for context in contexts:
        if context.chunk.source not in deduped:
            deduped.append(context.chunk.source)
    return deduped


def analyse_complementarity(
    pipeline: RAGPipeline, golden: list[dict[str, Any]], k: int
) -> dict[str, Any]:
    """Which questions each retriever gets that the other one misses.

    An aggregate score says whether fusion helped; this says *why*. If neither
    list has entries, the two retrievers are finding the same documents and
    fusing them is pure overhead - worth knowing before shipping the cost.
    """
    dense_only: list[dict[str, str]] = []
    lexical_only: list[dict[str, str]] = []
    both_miss: list[dict[str, str]] = []

    for record in golden:
        if not record["answerable"]:
            continue
        relevant = set(record["relevant_sources"])
        dense_hit = bool(relevant & set(retrieved_sources(pipeline, record["question"], k, ("dense",))))
        lexical_hit = bool(relevant & set(retrieved_sources(pipeline, record["question"], k, ("lexical",))))
        entry = {"id": record["id"], "question": record["question"]}
        if dense_hit and not lexical_hit:
            dense_only.append(entry)
        elif lexical_hit and not dense_hit:
            lexical_only.append(entry)
        elif not dense_hit and not lexical_hit:
            both_miss.append(entry)

    return {
        "dense_only": dense_only,
        "lexical_only": lexical_only,
        "both_miss": both_miss,
    }


def k_sweep(
    pipeline: RAGPipeline, golden: list[dict[str, Any]], ks: list[int]
) -> list[dict[str, Any]]:
    """Scores at several retrieval depths.

    A single k can flatter a retriever: one that saturates recall at k=5 looks
    unbeatable there while being clearly worse at k=1, where it matters most.
    """
    rows = []
    for k in ks:
        for name, sources in ABLATIONS.items():
            agg = evaluate_retrieval(pipeline, golden, k, sources)["aggregate"]
            rows.append(
                {
                    "k": k,
                    "retriever": name,
                    "recall": agg[f"recall@{k}"],
                    "mrr": agg["mrr"],
                    "ndcg": agg[f"ndcg@{k}"],
                }
            )
    return rows


def format_markdown(report: dict[str, Any]) -> str:
    k = report["k"]
    lines = [
        "# Evaluation results",
        "",
        f"- Corpus: **{report['stats']['documents']} documents**, "
        f"**{report['stats']['chunks']} chunks**",
        f"- Embeddings: `{report['stats']['embedder']}` "
        f"({report['stats']['embedding_dim']}d)",
        f"- Questions: **{report['questions']}** "
        f"({report['answerable']} answerable, "
        f"{report['questions'] - report['answerable']} out-of-domain)",
        f"- Retrieval depth: k = {k}",
        f"- Generated: {report['generated_at']}",
        "",
        "## Retrieval",
        "",
        f"| Retriever | Hit rate@{k} | Recall@{k} | Precision@{k} | nDCG@{k} | MRR |",
        "|---|---|---|---|---|---|",
    ]
    for name, result in report["retrieval"].items():
        agg = result["aggregate"]
        lines.append(
            f"| {name} | {agg[f'hit_rate@{k}']:.3f} | {agg[f'recall@{k}']:.3f} | "
            f"{agg[f'precision@{k}']:.3f} | {agg[f'ndcg@{k}']:.3f} | {agg['mrr']:.3f} |"
        )

    best = report["retrieval"][report["best_retriever"]]
    lines += ["", "### Recall by question type (best retriever)", "", "| Type | Recall |", "|---|---|"]
    for question_type, value in best["recall_by_type"].items():
        lines.append(f"| {question_type} | {value:.3f} |")

    if best["failures"]:
        lines += ["", "### Retrieval misses", "",
                  "Questions where no relevant document reached the top k.", ""]
        for failure in best["failures"]:
            lines.append(
                f"- **{failure['id']}** {failure['question']}  \n"
                f"  expected `{failure['expected']}`, got `{failure['retrieved']}`"
            )
    else:
        lines += ["", "No retrieval misses: every answerable question found a "
                  "relevant document within the top k."]

    if report.get("k_sweep"):
        lines += [
            "",
            "## Retrieval depth sweep",
            "",
            "A single k flatters whichever retriever happens to saturate there.",
            "",
            "| k | Retriever | Recall@k | MRR | nDCG@k |",
            "|---|---|---|---|---|",
        ]
        for row in report["k_sweep"]:
            lines.append(
                f"| {row['k']} | {row['retriever']} | {row['recall']:.3f} | "
                f"{row['mrr']:.3f} | {row['ndcg']:.3f} |"
            )

    if report.get("complementarity"):
        comp = report["complementarity"]
        lines += [
            "",
            f"## Retriever complementarity (k = {k})",
            "",
            f"- Found by dense but missed by BM25: **{len(comp['dense_only'])}**",
            f"- Found by BM25 but missed by dense: **{len(comp['lexical_only'])}**",
            f"- Missed by both: **{len(comp['both_miss'])}**",
            "",
        ]
        for label, key in [
            ("Dense-only wins", "dense_only"),
            ("BM25-only wins", "lexical_only"),
            ("Missed by both", "both_miss"),
        ]:
            if comp[key]:
                lines.append(f"**{label}**")
                lines.append("")
                for entry in comp[key]:
                    lines.append(f"- `{entry['id']}` {entry['question']}")
                lines.append("")

    if report.get("answers"):
        answers = report["answers"]
        lines += [
            "",
            "## Answer quality",
            "",
            f"- LLM backend: `{report['llm_backend']}`",
            f"- Keyword coverage: **{answers['keyword_coverage']:.3f}**",
            f"- Citation rate: **{answers['citation_rate']:.3f}**",
            f"- Abstention accuracy: **{answers['abstention_accuracy']:.3f}**",
            f"- p50 latency: {answers['p50_latency_ms']:.0f} ms",
        ]

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", type=int, default=5, help="Retrieval depth to score at")
    parser.add_argument("--ablate", action="store_true", help="Compare dense / BM25 / hybrid")
    parser.add_argument(
        "--k-sweep",
        type=str,
        default=None,
        help="Comma-separated depths to also score at, e.g. 1,3,5,10",
    )
    parser.add_argument("--generate", action="store_true", help="Also score generated answers")
    parser.add_argument("--llm-backend", choices=["gemini", "mock"], default=None)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "results.md")
    parser.add_argument("--json-out", type=Path, default=REPO_ROOT / "eval" / "results.json")
    args = parser.parse_args()

    overrides = {"llm_backend": args.llm_backend} if args.llm_backend else {}
    settings = get_settings(**overrides)
    pipeline = RAGPipeline.load(settings)
    golden = load_golden(args.golden)

    modes = ABLATIONS if args.ablate else {"hybrid (RRF)": ("dense", "lexical")}
    retrieval: dict[str, Any] = {}
    for name, sources in modes.items():
        started = time.perf_counter()
        retrieval[name] = evaluate_retrieval(pipeline, golden, args.k, sources)
        elapsed = time.perf_counter() - started
        agg = retrieval[name]["aggregate"]
        print(
            f"{name:<14} recall@{args.k}={agg[f'recall@{args.k}']:.3f} "
            f"mrr={agg['mrr']:.3f} ndcg={agg[f'ndcg@{args.k}']:.3f} ({elapsed:.1f}s)"
        )

    best_retriever = max(
        retrieval, key=lambda name: retrieval[name]["aggregate"][f"recall@{args.k}"]
    )

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "k": args.k,
        "stats": pipeline.stats(),
        "questions": len(golden),
        "answerable": sum(1 for r in golden if r["answerable"]),
        "retrieval": retrieval,
        "best_retriever": best_retriever,
        "llm_backend": settings.llm_backend,
    }

    if args.ablate:
        report["complementarity"] = analyse_complementarity(pipeline, golden, args.k)
        comp = report["complementarity"]
        print(
            f"\ncomplementarity@{args.k}: dense-only={len(comp['dense_only'])} "
            f"bm25-only={len(comp['lexical_only'])} both-miss={len(comp['both_miss'])}"
        )

    if args.k_sweep:
        ks = [int(part) for part in args.k_sweep.split(",") if part.strip()]
        report["k_sweep"] = k_sweep(pipeline, golden, ks)

    if args.generate:
        print("\nGenerating answers...")
        report["answers"] = evaluate_answers(pipeline, golden, args.k)
        answers = report["answers"]
        print(
            f"keyword_coverage={answers['keyword_coverage']:.3f} "
            f"citation_rate={answers['citation_rate']:.3f} "
            f"abstention_accuracy={answers['abstention_accuracy']:.3f}"
        )

    args.out.write_text(format_markdown(report), encoding="utf-8")
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out} and {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
