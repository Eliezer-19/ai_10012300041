"""
Cross-encoder re-ranking for hybrid retrieval (Part B / Part D).

Scores (query, passage) pairs with a small MS MARCO–style cross-encoder and re-sorts hits.
Heavier than bi-encoder retrieval but typically improves top-k precision vs. only increasing k.

Student: Eliezer Anim-Somuah · Index: 10012300041
"""

from __future__ import annotations

from functools import lru_cache

from part_b.retrieval_system import RetrievedChunk


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def cross_encoder_rerank(
    query: str,
    hits: list[RetrievedChunk],
    model_name: str,
    *,
    top_k: int | None = None,
    batch_size: int = 16,
) -> list[RetrievedChunk]:
    """
    Re-score and sort `hits` by cross-encoder relevance. Optionally keep only the first `top_k`.

    Preserves chunk fields; replaces ``score`` / ``score_kind`` and records prior scores in ``meta``.
    """
    if not hits:
        return hits
    q = query.strip()
    if not q:
        return hits if top_k is None else hits[:top_k]

    ce = _load_cross_encoder(model_name)
    pairs = [[q, h.text] for h in hits]
    raw = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    scores = [float(s) for s in raw]

    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    out: list[RetrievedChunk] = []
    for i in order:
        h = hits[i]
        s = scores[i]
        meta = dict(h.meta)
        meta["pre_rerank_score"] = h.score
        meta["pre_rerank_kind"] = h.score_kind
        out.append(
            RetrievedChunk(
                chunk_id=h.chunk_id,
                source=h.source,
                doc_type=h.doc_type,
                text=h.text,
                meta=meta,
                score=s,
                score_kind="cross_encoder",
            )
        )
    if top_k is not None:
        out = out[:top_k]
    return out
