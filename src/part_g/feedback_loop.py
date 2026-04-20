"""
Part G — Feedback loop for improving retrieval.

Student: Eliezer Anim-Somuah · Index: 10012300041

Stores (query text, chunk_id, label) where label is +1 (relevant/helpful) or -1 (not helpful).
At query time, past feedback whose *query embedding* is similar to the current query adjusts
scores for those chunk_ids, then hits are re-sorted. This implements a lightweight
**embedding-similarity prior** over hybrid RRF results without retraining the index.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from part_b.retrieval_system import EmbeddingPipeline, RetrievedChunk


@dataclass
class FeedbackRow:
    query: str
    chunk_id: str
    label: int  # +1 or -1


class FeedbackStore:
    """JSONL append-only store: one JSON object per line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._emb_cache: dict[str, np.ndarray] = {}

    def append(self, query: str, chunk_id: str, label: int) -> None:
        if label not in (-1, 1):
            raise ValueError("label must be -1 or +1")
        q = query.strip()
        if not q or not chunk_id.strip():
            raise ValueError("query and chunk_id must be non-empty")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"query": q, "chunk_id": chunk_id.strip(), "label": label}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._emb_cache.pop(q, None)

    def load_rows(self) -> list[FeedbackRow]:
        if not self.path.is_file():
            return []
        out: list[FeedbackRow] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append(
                    FeedbackRow(
                        query=str(d["query"]),
                        chunk_id=str(d["chunk_id"]),
                        label=int(d["label"]),
                    )
                )
        return out

    def _embed_query(self, q: str, embedder: EmbeddingPipeline) -> np.ndarray:
        if q not in self._emb_cache:
            self._emb_cache[q] = embedder.encode_query(q).astype(np.float32).ravel()
        return self._emb_cache[q]

    def chunk_boosts(
        self,
        query: str,
        embedder: EmbeddingPipeline,
        *,
        min_sim: float = 0.68,
        weight: float = 0.25,
    ) -> dict[str, float]:
        """
        Aggregate boost per chunk_id from feedback rows whose stored query is
        semantically close to `query` (cosine similarity on normalized embeddings).
        boost += weight * label * sim
        """
        rows = self.load_rows()
        if not rows:
            return {}
        qv = embedder.encode_query(query).astype(np.float32).ravel()
        boosts: dict[str, float] = defaultdict(float)
        for r in rows:
            fv = self._embed_query(r.query, embedder)
            sim = float(np.dot(qv, fv))
            if sim < min_sim:
                continue
            boosts[r.chunk_id] += weight * float(r.label) * sim
        return dict(boosts)


def apply_feedback_rerank(
    hits: list[RetrievedChunk],
    query: str,
    embedder: EmbeddingPipeline,
    store: FeedbackStore | Path | None,
    *,
    min_sim: float = 0.68,
    weight: float = 0.25,
) -> list[RetrievedChunk]:
    """
    Re-rank `hits` by: new_score = base_score + boost(chunk_id).
    `score_kind` becomes `hybrid_rrf_feedback` when any hit receives a non-zero boost.
    """
    if not store or not hits:
        return hits
    fs = store if isinstance(store, FeedbackStore) else FeedbackStore(store)
    boosts = fs.chunk_boosts(query, embedder, min_sim=min_sim, weight=weight)
    if not boosts:
        return hits
    if not any(h.chunk_id in boosts for h in hits):
        return hits

    adjusted: list[RetrievedChunk] = []
    touched = False
    for h in hits:
        b = boosts.get(h.chunk_id, 0.0)
        if b != 0.0:
            touched = True
        meta = dict(h.meta)
        if b != 0.0:
            meta["feedback_boost"] = b
        adjusted.append(
            RetrievedChunk(
                chunk_id=h.chunk_id,
                source=h.source,
                doc_type=h.doc_type,
                text=h.text,
                meta=meta,
                score=float(h.score) + b,
                score_kind=h.score_kind,
            )
        )
    adjusted.sort(key=lambda x: x.score, reverse=True)
    if not touched:
        return hits
    return [replace(h, score_kind="hybrid_rrf_feedback") for h in adjusted]
