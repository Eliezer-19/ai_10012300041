"""
PART B — Custom retrieval: embeddings + FAISS + hybrid (keyword + vector) search.

Student: Eliezer Anim-Somuah · Index: 10012300041

Components:
  - Embedding pipeline: sentence-transformers (normalized vectors → cosine via inner product)
  - Vector storage: FAISS IndexFlatIP (in-memory; persist optional)
  - Top-k retrieval with similarity scores
  - Hybrid search: combine dense retrieval with TF-IDF keyword scores (RRF fusion)
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    doc_type: str
    text: str
    meta: dict[str, Any]
    score: float
    score_kind: str


def load_chunks_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class EmbeddingPipeline:
    """Encode queries and passages with a sentence-transformer model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 500,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)


class FaissVectorStore:

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._built = False

    def add(self, vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected ({self.dim},) per row, got {vectors.shape}")
        self.index.add(vectors)
        self._built = True

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, indices = self.index.search(query_vec, k)
        return scores[0], indices[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path))
        self._built = True


class KeywordIndex:
    """Keyword index for hybrid search."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        self._matrix = None

    def fit(self, corpus: list[str]) -> None:
        self._matrix = self.vectorizer.fit_transform(corpus)

    def query_scores(self, query: str) -> np.ndarray:
        if self._matrix is None:
            raise RuntimeError("fit() first")
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix).ravel()
        return sims.astype(np.float32)


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k_rrf: int = 60,
) -> dict[int, float]:
    """RRF: score(i) = sum_j 1/(k_rrf + rank_j(i)). Missing docs omitted."""
    scores: dict[int, float] = {}
    for ids in ranked_lists:
        for rank, idx in enumerate(ids, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank)
    return scores


def _digit_runs_from_query(query: str, min_len: int = 4) -> list[str]:
    """Long digit runs (e.g. vote totals) for exact-ish matching despite commas in text."""
    return [m.group(0) for m in re.finditer(rf"\d{{{min_len},}}", query)]


def _chunk_contains_digit_run(text: str, run: str) -> bool:
    digits = re.sub(r"\D", "", text)
    return run in digits


class HybridRetriever:
    """Dense (FAISS) + lexical (TF-IDF) hybrid via Reciprocal Rank Fusion."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embedder: EmbeddingPipeline,
        faiss_store: FaissVectorStore,
        keyword_index: KeywordIndex,
        passage_embeddings: np.ndarray,
    ) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.faiss_store = faiss_store
        self.keyword_index = keyword_index
        self._passage_embeddings = passage_embeddings

    @classmethod
    def build(
        cls,
        chunks: list[dict[str, Any]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> HybridRetriever:
        texts = [c["text"] for c in chunks]
        embedder = EmbeddingPipeline(model_name=model_name)
        passage_embeddings = embedder.encode_passages(texts)
        store = FaissVectorStore(embedder.dim)
        store.add(passage_embeddings)

        kw = KeywordIndex()
        kw.fit(texts)

        return cls(
            chunks=chunks,
            embedder=embedder,
            faiss_store=store,
            keyword_index=kw,
            passage_embeddings=passage_embeddings,
        )

    def retrieve_dense(self, query: str, k: int) -> list[RetrievedChunk]:
        qv = self.embedder.encode_query(query)
        scores, indices = self.faiss_store.search(qv, k)
        out: list[RetrievedChunk] = []
        for rank, (idx, sc) in enumerate(zip(indices.tolist(), scores.tolist(), strict=True)):
            if idx < 0:
                continue
            c = self.chunks[idx]
            out.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    source=c["source"],
                    doc_type=c["doc_type"],
                    text=c["text"],
                    meta=c.get("meta", {}),
                    score=float(sc),
                    score_kind="cosine_ip",
                )
            )
        return out

    def retrieve_keyword_top_indices(self, query: str, k: int) -> list[int]:
        sims = self.keyword_index.query_scores(query)
        top = np.argsort(-sims)[:k]
        return top.tolist()

    def retrieve_hybrid_rrf(
        self,
        query: str,
        k: int,
        *,
        candidate_pool: int = 80,
        k_rrf: int = 60,
        numeric_boost: bool = True,
        numeric_boost_weight: float = 2.0,
    ) -> list[RetrievedChunk]:
        """RRF over dense top-N and lexical top-N; return top-k fused."""
        qv = self.embedder.encode_query(query)
        _, dense_idx = self.faiss_store.search(qv, min(candidate_pool, len(self.chunks)))
        dense_order = [i for i in dense_idx.tolist() if i >= 0][:candidate_pool]

        lex_order = self.retrieve_keyword_top_indices(query, min(candidate_pool, len(self.chunks)))

        rrf_scores = reciprocal_rank_fusion([dense_order, lex_order], k_rrf=k_rrf)
        runs = _digit_runs_from_query(query) if numeric_boost else []
        if runs:
            digit_prior = 1.0 / (k_rrf + 250)
            for i, ch in enumerate(self.chunks):
                if any(_chunk_contains_digit_run(ch["text"], r) for r in runs):
                    rrf_scores[i] = rrf_scores.get(i, 0.0) + digit_prior

        if runs:
            boosted: dict[int, float] = {}
            for doc_i in rrf_scores.keys():
                base = float(rrf_scores[doc_i])
                text = self.chunks[doc_i]["text"]
                frac = sum(1 for r in runs if _chunk_contains_digit_run(text, r)) / len(runs)
                boosted[doc_i] = base * (1.0 + numeric_boost_weight * frac)
            sorted_ids = sorted(boosted.keys(), key=lambda i: boosted[i], reverse=True)[:k]
            score_map = boosted
            kind = "hybrid_rrf_numeric"
        else:
            sorted_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:k]
            score_map = {i: float(rrf_scores[i]) for i in sorted_ids}
            kind = "hybrid_rrf"

        out: list[RetrievedChunk] = []
        for doc_i in sorted_ids:
            c = self.chunks[doc_i]
            out.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    source=c["source"],
                    doc_type=c["doc_type"],
                    text=c["text"],
                    meta=c.get("meta", {}),
                    score=float(score_map[doc_i]),
                    score_kind=kind,
                )
            )
        return out


def persist_bundle(
    out_dir: Path,
    retriever: HybridRetriever,
    model_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": model_name,
        "num_chunks": len(retriever.chunks),
        "chunk_ids": [c["chunk_id"] for c in retriever.chunks],
    }
    (out_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    retriever.faiss_store.save(out_dir / "faiss.index")
    np.save(out_dir / "passage_embeddings.npy", retriever._passage_embeddings)
    with (out_dir / "chunks.pkl").open("wb") as f:
        pickle.dump(retriever.chunks, f)
    with (out_dir / "keyword_vectorizer.pkl").open("wb") as f:
        pickle.dump(retriever.keyword_index.vectorizer, f)
        pickle.dump(retriever.keyword_index._matrix, f)


def load_bundle(out_dir: Path, model_name: str | None = None) -> HybridRetriever:
    meta = json.loads((out_dir / "index_meta.json").read_text(encoding="utf-8"))
    mn = model_name or meta["model_name"]
    chunks = pickle.loads((out_dir / "chunks.pkl").read_bytes())
    passage_embeddings = np.load(out_dir / "passage_embeddings.npy")

    embedder = EmbeddingPipeline(model_name=mn)
    store = FaissVectorStore(embedder.dim)
    store.load(out_dir / "faiss.index")

    with (out_dir / "keyword_vectorizer.pkl").open("rb") as f:
        vectorizer = pickle.load(f)
        matrix = pickle.load(f)
    kw = KeywordIndex()
    kw.vectorizer = vectorizer
    kw._matrix = matrix

    return HybridRetriever(
        chunks=chunks,
        embedder=embedder,
        faiss_store=store,
        keyword_index=kw,
        passage_embeddings=passage_embeddings,
    )
