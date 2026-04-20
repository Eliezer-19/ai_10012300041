"""
Append one relevance label to the feedback store (Part G).

Student: Eliezer Anim-Somuah · Index: 10012300041

Examples (from repository root):
  python src/part_g/record_feedback.py --query "VAT measures" --chunk-id "2025-Budget-...:para:42" --positive
  python src/part_g/record_feedback.py --query "VAT measures" --chunk-id "2025-Budget-...:para:99" --negative
  python src/part_g/record_feedback.py --store outputs/part_g/my_feedback.jsonl ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from academic_city.constants import DEFAULT_FEEDBACK_STORE
from part_g.feedback_loop import FeedbackStore


def main() -> None:
    ap = argparse.ArgumentParser(description="Record retrieval feedback (+1 / -1) for feedback-augmented RAG")
    ap.add_argument("--query", required=True, help="The query text used when the chunk was shown")
    ap.add_argument("--chunk-id", required=True, dest="chunk_id", help="Exact chunk_id from the RAG report")
    ap.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_FEEDBACK_STORE,
        help="JSONL feedback file (default: outputs/part_g/feedback.jsonl)",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--positive", action="store_true", help="Chunk was relevant / helpful")
    g.add_argument("--negative", action="store_true", help="Chunk was irrelevant or harmful")
    args = ap.parse_args()

    label = 1 if args.positive else -1
    store = FeedbackStore(args.store)
    store.append(args.query, args.chunk_id, label)
    print(f"Recorded label={label:+d} for chunk_id={args.chunk_id!r}", file=sys.stderr)
    print(f"Appended to {args.store}", file=sys.stderr)


if __name__ == "__main__":
    main()
