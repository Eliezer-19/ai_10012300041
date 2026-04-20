"""
PART C — Context window management for RAG prompts.

Student: Eliezer Anim-Somuah · Index: 10012300041

Strategies:
  - Rank by retrieval score (already sorted from hybrid retriever)
  - Filter by minimum score (drop noisy tails)
  - Truncate per chunk and total budget to fit model context
"""

from __future__ import annotations

from dataclasses import dataclass, field

from part_b.retrieval_system import RetrievedChunk


@dataclass
class PackedContext:
    """Formatted context blocks for injection into prompts."""

    numbered_text: str
    """Blocks like [1] (score=0.12 | budget_pdf) ... text ..."""
    meta: dict = field(default_factory=dict)
    blocks_used: int = 0
    chars_total: int = 0
    dropped_count: int = 0


def pack_context_for_prompt(
    hits: list[RetrievedChunk],
    *,
    max_total_chars: int = 12_000,
    max_chunk_chars: int = 2_800,
    min_score: float | None = None,
    include_meta_line: bool = True,
) -> PackedContext:
    """
    Build a single string of numbered passages, respecting a global char budget.

    - **Rank**: preserves order of `hits` (caller should pass hybrid/dense results sorted by score).
    - **Filter**: drops chunks with score < min_score when set.
    - **Truncate**: each passage trimmed to `max_chunk_chars`; stop adding when `max_total_chars` reached.
    """
    parts: list[str] = []
    total = 0
    score_skipped = 0

    for h in hits:
        if min_score is not None and h.score < min_score:
            score_skipped += 1
            continue
        body = h.text.strip()
        if len(body) > max_chunk_chars:
            body = body[: max_chunk_chars - 3] + "..."
        idx = len(parts) + 1
        header = f"[{idx}]"
        if include_meta_line:
            header += f" (retrieval_score={h.score:.4f} | {h.doc_type} | {h.source})"
        block = f"{header}\n{body}"
        if total + len(block) + 2 > max_total_chars:
            break
        parts.append(block)
        total += len(block) + 2

    not_included_after_budget = max(0, len(hits) - score_skipped - len(parts))

    if not parts:
        return PackedContext(
            numbered_text="(no passages selected — widen budget or lower min_score)",
            meta={"warning": "empty"},
            blocks_used=0,
            chars_total=0,
            dropped_count=score_skipped + not_included_after_budget,
        )

    numbered_text = "\n\n".join(parts)
    return PackedContext(
        numbered_text=numbered_text,
        meta={
            "max_total_chars": max_total_chars,
            "max_chunk_chars": max_chunk_chars,
            "min_score": min_score,
            "score_filtered": score_skipped,
            "budget_truncated_rest": not_included_after_budget,
        },
        blocks_used=len(parts),
        chars_total=len(numbered_text),
        dropped_count=score_skipped + not_included_after_budget,
    )
