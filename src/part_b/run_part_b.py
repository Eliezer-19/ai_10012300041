"""
PART B — Build index, run retrieval demos, document failure cases + hybrid fix.

Student: Eliezer Anim-Somuah · Index: 10012300041

Usage (from repository root):
  pip install -e .
  python src/part_b/run_part_b.py              # build from Part A chunks + save to outputs/part_b/
  python src/part_b/run_part_b.py --load     # load cached index from outputs/part_b/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from academic_city.constants import DEFAULT_CHUNKS_JSONL, PART_B_OUTPUT_DIR
from academic_city.text_utils import preview_text
from part_b.retrieval_system import HybridRetriever, load_bundle, load_chunks_jsonl, persist_bundle

CHUNKS_DEFAULT = DEFAULT_CHUNKS_JSONL
OUT_DIR = PART_B_OUTPUT_DIR

# Queries chosen to expose pure-dense failures (semantic drift, wrong chunk, or exact-number mismatch).
FAILURE_QUERIES: list[dict[str, str]] = [
    {
        "id": "q1",
        "query": "Public Financial Management Act 2016 Act 921 Section 28 budget presentation",
        "why_fail": "Dense retrieval often ranks broadly similar policy paragraphs first; the chunk that actually cites Act 921 for presenting the budget can sit lower in the ranking.",
        "fix": "Hybrid RRF: the TF‑IDF leg strongly matches rare tokens ('921', 'Section 28', 'Public Financial Management Act') and promotes the correct passage.",
    },
    {
        "id": "q2",
        "query": "946048 votes Greater Accra",
        "why_fail": "Cosine similarity is not reliable for exact vote totals: semantically similar 'Greater Accra' election rows (other years/candidates) can outrank the row containing 946,048.",
        "fix": "After RRF, a numeric-aware stage boosts chunks whose text contains the query’s digit run (comma-insensitive), so the 2016 Mahama row surfaces.",
    },
    {
        "id": "q3",
        "query": "John Dramani Mahama votes percentage Greater Accra 2016",
        "why_fail": "Many rows share the same candidate + year embeddingally; dense top-k can mix regions before the user’s intended row.",
        "fix": "Hybrid search rewards co-occurrence of '2016', 'Greater Accra', and the candidate name via the lexical leg.",
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=CHUNKS_DEFAULT)
    ap.add_argument("--load", action="store_true", help="Load cached index from outputs/part_b/")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    if args.load and (OUT_DIR / "faiss.index").exists():
        retriever = load_bundle(OUT_DIR, model_name=args.model)
        print("Loaded index from", OUT_DIR)
    else:
        if not args.chunks.exists():
            raise FileNotFoundError(f"Chunks not found: {args.chunks}. Run part_a first.")
        chunks = load_chunks_jsonl(args.chunks)
        print(f"Building hybrid retriever over {len(chunks)} chunks...")
        retriever = HybridRetriever.build(chunks, model_name=args.model)
        persist_bundle(OUT_DIR, retriever, model_name=args.model)
        print("Saved index to", OUT_DIR)

    report: list[str] = []
    report.append("# PART B — Retrieval failures vs hybrid fix\n\n")
    report.append("**Setup**: Sentence Transformers embeddings + FAISS (inner product on normalized vectors = cosine similarity). ")
    report.append(
        "**Extension**: **Hybrid search** — Reciprocal Rank Fusion (RRF) of dense top-N and TF-IDF top-N; "
        "queries with long digit runs (e.g. vote totals) additionally use a **numeric-aware** re-score.\n\n"
    )

    for item in FAILURE_QUERIES:
        q = item["query"]
        report.append(f"## {item['id']}: {q[:80]}...\n\n")
        report.append(f"- **Why vector-only can fail**: {item['why_fail']}\n")
        report.append(f"- **Fix**: {item['fix']}\n\n")

        dense = retriever.retrieve_dense(q, args.topk)
        hybrid = retriever.retrieve_hybrid_rrf(q, args.topk)

        report.append("### Dense-only (top-k)\n\n")
        for i, h in enumerate(dense, 1):
            report.append(f"{i}. score=`{h.score:.4f}` | {h.doc_type} | {preview_text(h.text, max_chars=280)}\n")
        report.append("\n### Hybrid (RRF + numeric when applicable) (top-k)\n\n")
        for i, h in enumerate(hybrid, 1):
            report.append(f"{i}. score=`{h.score:.4f}` | {h.doc_type} | {preview_text(h.text, max_chars=280)}\n")
        report.append("\n---\n\n")

        if item["id"] in ("q2", "q3"):
            d_top = dense[0].doc_type if dense else ""
            h_top = hybrid[0].doc_type if hybrid else ""
            report.append(f"*Doc-type check*: dense top-1=`{d_top}` → hybrid top-1=`{h_top}`.\n\n")

    (OUT_DIR / "PART_B_FAILURES_AND_FIX.md").write_text("".join(report), encoding="utf-8")
    print("Wrote", OUT_DIR / "PART_B_FAILURES_AND_FIX.md")

    # Console summary
    print("\n--- Sample: first failure query ---\n")
    q0 = FAILURE_QUERIES[0]["query"]
    print("Query:", q0)
    print("\nDense top-3:")
    for h in retriever.retrieve_dense(q0, 3):
        print(f"  {h.score:.4f} [{h.doc_type}]", preview_text(h.text, max_chars=120))
    print("\nHybrid top-3:")
    for h in retriever.retrieve_hybrid_rrf(q0, 3):
        print(f"  {h.score:.4f} [{h.doc_type}]", preview_text(h.text, max_chars=120))


if __name__ == "__main__":
    main()
