"""
PART A — Data Engineering & Preparation

Student: Eliezer Anim-Somuah · Index: 10012300041

Inputs:
  - ../2025-Budget-Statement-and-Economic-Policy_v4.pdf
  - ../Ghana_Election_Result.csv

Outputs (written to ``outputs/part_a/`` at repo root):
  - cleaned_budget_text.txt
  - cleaned_election_rows.csv
  - chunks_<strategy>.jsonl
  - chunking_comparison.md
  
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import re
import statistics
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from academic_city.constants import PART_A_OUTPUT_DIR
from academic_city.paths import project_root

ROOT = project_root()
OUT_DIR = PART_A_OUTPUT_DIR
PDF_PATH = ROOT / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
CSV_PATH = ROOT / "Ghana_Election_Result.csv"


def _norm_ws(s: str) -> str:
    s = s.replace("\u00a0", " ")  # NBSP
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _strip_page_artifacts(s: str) -> str:
    # Remove common PDF-to-text artifacts we observed in the extracted preview
    s = re.sub(r"--\s*\d+\s+of\s+\d+\s*--", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n\s*\d+\s*\n", "\n", s)  # stray page numbers on their own line
    s = re.sub(r"\.{3,}", " ", s)  # dotted leaders in TOC/list-of-tables
    return _norm_ws(s)


def extract_and_clean_budget_pdf(pdf_path: Path) -> dict[str, Any]:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        txt = _strip_page_artifacts(txt)
        if txt:
            pages.append(f"[PAGE {i}]\n{txt}")

    full_text = "\n\n".join(pages)
    full_text = _norm_ws(full_text)

    # Keep a light structural normalization of section headings
    # Example: "SECTION 1: INTRODUCTION"
    full_text = re.sub(r"(SECTION\s+\d+\s*:\s*[A-Z][A-Z '\-&]+)", r"\n\n\1\n", full_text)
    full_text = _norm_ws(full_text)

    return {
        "source": str(pdf_path.name),
        "page_count": len(reader.pages),
        "text": full_text,
    }


def load_and_clean_election_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().replace("\u00a0", " ") for c in df.columns]

    # Normalize whitespace in key string columns
    for col in ["Old Region", "New Region", "Code", "Candidate", "Party"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: _norm_ws(x))

    # Standardize "Code" to 3 groups: NPP, NDC, OTHERS (csv contains "Others"/"OTHERS")
    if "Code" in df.columns:
        df["Code"] = df["Code"].str.upper()
        df["Code"] = df["Code"].replace({"OTHER": "OTHERS", "OTHERS": "OTHERS"})

    # Votes as int
    if "Votes" in df.columns:
        df["Votes"] = (
            df["Votes"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").astype("Int64")

    # Votes(%) to float in [0, 100]
    pct_col = None
    for c in df.columns:
        if c.lower().startswith("votes("):
            pct_col = c
            break
    if pct_col:
        df[pct_col] = (
            df[pct_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")

    # Drop rows with missing essential fields
    essential = ["Year", "New Region", "Candidate", "Party", "Votes"]
    for c in essential:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df = df.dropna(subset=["Year", "New Region", "Candidate", "Party", "Votes"]).copy()

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    # Sort for stable downstream grouping/text generation
    df = df.sort_values(["Year", "New Region", "Party", "Candidate"], kind="mergesort").reset_index(drop=True)
    return df


def election_rows_to_text_docs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Turn a tabular dataset into retrieval-friendly text documents.
    For RAG, we want natural-language statements that preserve key entities.
    """
    pct_col = next((c for c in df.columns if c.lower().startswith("votes(")), None)

    docs: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        year = int(row["Year"])
        old_region = str(row.get("Old Region", "")).strip()
        new_region = str(row["New Region"]).strip()
        candidate = str(row["Candidate"]).strip()
        party = str(row["Party"]).strip()
        code = str(row.get("Code", "")).strip()
        votes = row["Votes"]
        votes_str = f"{int(votes):,}" if pd.notna(votes) else "N/A"

        pct_str = ""
        if pct_col and pd.notna(row[pct_col]):
            pct_val = float(row[pct_col])
            pct_str = f" ({pct_val:.2f}%)"

        region_note = f" (old region: {old_region})" if old_region and old_region != new_region else ""
        text = (
            f"Ghana presidential election results: In {year}, in {new_region}{region_note}, "
            f"{candidate} ({party}; code: {code}) received {votes_str} votes{pct_str}."
        )
        docs.append(
            {
                "source": "Ghana_Election_Result.csv",
                "doc_type": "election_row",
                "row_index": int(row_idx),
                "year": year,
                "region": new_region,
                "candidate": candidate,
                "party": party,
                "text": text,
            }
        )
    return docs


@dataclasses.dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    doc_type: str
    text: str
    meta: dict[str, Any]


def chunk_fixed_chars(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    source: str,
    doc_type: str,
    base_meta: dict[str, Any] | None = None,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")
    base_meta = dict(base_meta or {})

    chunks: list[Chunk] = []
    start = 0
    n = len(text)
    k = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{source}:{doc_type}:char:{k}",
                    source=source,
                    doc_type=doc_type,
                    text=chunk_text,
                    meta={**base_meta, "start_char": start, "end_char": end, "chunk_size": chunk_size, "overlap": overlap},
                )
            )
            k += 1
        if end == n:
            break
        start = end - overlap
    return chunks


def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


def chunk_paragraph_packed(
    text: str,
    *,
    max_chars: int,
    overlap_paragraphs: int,
    source: str,
    doc_type: str,
    base_meta: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Structure-aware chunking: keep paragraphs together, pack into ~max_chars, then overlap by paragraphs.
    This tends to preserve coherence better than raw character windows for narrative PDFs.
    """
    base_meta = dict(base_meta or {})
    paras = _split_paragraphs(text)
    if not paras:
        return []

    chunks: list[Chunk] = []
    i = 0
    k = 0
    while i < len(paras):
        cur: list[str] = []
        cur_len = 0
        start_i = i
        while i < len(paras):
            p = paras[i]
            add_len = len(p) + (2 if cur else 0)
            if cur and cur_len + add_len > max_chars:
                break
            cur.append(p)
            cur_len += add_len
            i += 1

        chunk_text = "\n\n".join(cur).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{source}:{doc_type}:para:{k}",
                    source=source,
                    doc_type=doc_type,
                    text=chunk_text,
                    meta={**base_meta, "start_paragraph": start_i, "end_paragraph": i, "max_chars": max_chars, "overlap_paragraphs": overlap_paragraphs},
                )
            )
            k += 1

        if i >= len(paras):
            break
        i = max(start_i + 1, i - overlap_paragraphs)
    return chunks


def chunk_fixed_words(
    text: str,
    *,
    words_per_chunk: int,
    overlap_words: int,
    source: str,
    doc_type: str,
    base_meta: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Token/word-based chunking: approximates model token windows without needing a tokenizer.
    Useful when you want chunk sizes that track semantic capacity more consistently than characters.
    """
    if overlap_words >= words_per_chunk:
        raise ValueError("overlap_words must be < words_per_chunk")
    base_meta = dict(base_meta or {})

    words = re.findall(r"\S+", text)
    chunks: list[Chunk] = []
    start = 0
    k = 0
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{source}:{doc_type}:word:{k}",
                    source=source,
                    doc_type=doc_type,
                    text=chunk_text,
                    meta={**base_meta, "start_word": start, "end_word": end, "words_per_chunk": words_per_chunk, "overlap_words": overlap_words},
                )
            )
            k += 1
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


def build_tfidf_retriever(chunks: list[Chunk]) -> tuple[TfidfVectorizer, Any]:
    corpus = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


def retrieve_topk(
    query: str,
    *,
    vectorizer: TfidfVectorizer,
    X: Any,
    chunks: list[Chunk],
    k: int,
) -> list[tuple[Chunk, float]]:
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).ravel()
    top_idx = sims.argsort()[::-1][:k]
    return [(chunks[i], float(sims[i])) for i in top_idx]


def _keywords_present(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return all(kw.lower() in t for kw in keywords)


def evaluate_chunking(
    chunks: list[Chunk],
    *,
    queries: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, Any]:
    vectorizer, X = build_tfidf_retriever(chunks)

    hits = 0
    rr: list[float] = []
    per_query: list[dict[str, Any]] = []
    for q in queries:
        query = q["query"]
        keywords = q["keywords"]
        ranked = retrieve_topk(query, vectorizer=vectorizer, X=X, chunks=chunks, k=k)

        hit_rank = None
        for rank, (ch, _score) in enumerate(ranked, start=1):
            if _keywords_present(ch.text, keywords):
                hit_rank = rank
                break

        if hit_rank is not None:
            hits += 1
            rr.append(1.0 / hit_rank)
        else:
            rr.append(0.0)

        per_query.append(
            {
                "query": query,
                "keywords": keywords,
                "hit_rank": hit_rank,
                "top_chunk_preview": (ranked[0][0].text[:220] + "...") if ranked else "",
            }
        )

    return {
        "num_chunks": len(chunks),
        "avg_chunk_chars": statistics.mean(len(c.text) for c in chunks) if chunks else 0,
        "hit_at_k": hits / max(1, len(queries)),
        "mrr_at_k": statistics.mean(rr) if rr else 0,
        "per_query": per_query,
    }


def write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": ch.chunk_id,
                        "source": ch.source,
                        "doc_type": ch.doc_type,
                        "text": ch.text,
                        "meta": ch.meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------
    # 1) Cleaning
    # --------------------
    budget = extract_and_clean_budget_pdf(PDF_PATH)
    (OUT_DIR / "cleaned_budget_text.txt").write_text(budget["text"], encoding="utf-8")

    election_df = load_and_clean_election_csv(CSV_PATH)
    election_df.to_csv(OUT_DIR / "cleaned_election_rows.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    election_docs = election_rows_to_text_docs(election_df)

    # Unified corpus: budget is long narrative; election is many small atomic rows.
    corpus_docs: list[dict[str, Any]] = [
        {"source": budget["source"], "doc_type": "budget_pdf", "text": budget["text"], "meta": {"page_count": budget["page_count"]}},
        *[
            {"source": d["source"], "doc_type": d["doc_type"], "text": d["text"], "meta": {k: v for k, v in d.items() if k not in {"source", "doc_type", "text"}}}
            for d in election_docs
        ],
    ]

    # --------------------
    # 2) Chunking strategies
    # --------------------
    # Strategy A: Fixed character windows (fast, simple baseline)
    # Strategy B: Word windows (more stable "semantic capacity" than chars)
    # Strategy C: Paragraph-packed (preserves narrative coherence for PDFs)

    chunks_a: list[Chunk] = []
    chunks_b: list[Chunk] = []
    chunks_c: list[Chunk] = []

    for doc in corpus_docs:
        source = doc["source"]
        doc_type = doc["doc_type"]
        text = doc["text"]
        meta = dict(doc.get("meta", {}))

        if doc_type == "budget_pdf":
            # PDF: chunk as longer narrative
            chunks_a.extend(chunk_fixed_chars(text, chunk_size=1200, overlap=200, source=source, doc_type=doc_type, base_meta=meta))
            chunks_b.extend(chunk_fixed_words(text, words_per_chunk=260, overlap_words=50, source=source, doc_type=doc_type, base_meta=meta))
            chunks_c.extend(chunk_paragraph_packed(text, max_chars=1400, overlap_paragraphs=1, source=source, doc_type=doc_type, base_meta=meta))
        else:
            # Election row docs are already small; we keep them as one "chunk" each.
            base = Chunk(
                chunk_id=f"{source}:{doc_type}:row:{meta.get('row_index','')}",
                source=source,
                doc_type=doc_type,
                text=text,
                meta=meta,
            )
            chunks_a.append(base)
            chunks_b.append(base)
            chunks_c.append(base)

    # --------------------
    # 3) Comparative retrieval analysis
    # --------------------
    # Note: Ground truth here is keyword-based. It’s lightweight but consistent and
    # good enough to show how chunk boundaries affect retrievability.
    queries = [
        # Budget-focused queries
        {"query": "What is the theme of the 2025 budget statement?", "keywords": ["theme", "resetting the economy"]},
        {"query": "Which section is about global economic developments and outlook?", "keywords": ["section 2", "global economic developments"]},
        {"query": "Where can electronic copies of the 2025 budget be downloaded from?", "keywords": ["electronic copies", "mofep.gov.gh"]},
        {"query": "Which law is the 2025 budget statement presented in accordance with?", "keywords": ["public financial management act", "act 921"]},
        # Election-focused queries
        {"query": "In 2020, how many votes did Nana Akufo Addo receive in Ashanti Region?", "keywords": ["2020", "ashanti", "nana akufo addo", "votes"]},
        {"query": "In 2016, what percentage did John Dramani Mahama get in Greater Accra Region?", "keywords": ["2016", "greater accra", "john dramani mahama", "%"]},
        {"query": "In 2012, who received votes in Volta Region for NPP?", "keywords": ["2012", "volta", "npp", "votes"]},
        {"query": "In 2008, what were the results in Upper West Region for NDC?", "keywords": ["2008", "upper west", "ndc", "votes"]},
    ]

    res_a = evaluate_chunking(chunks_a, queries=queries, k=5)
    res_b = evaluate_chunking(chunks_b, queries=queries, k=5)
    res_c = evaluate_chunking(chunks_c, queries=queries, k=5)

    # Write chunk datasets
    write_jsonl(OUT_DIR / "chunks_fixed_chars.jsonl", chunks_a)
    write_jsonl(OUT_DIR / "chunks_fixed_words.jsonl", chunks_b)
    write_jsonl(OUT_DIR / "chunks_paragraph_packed.jsonl", chunks_c)

    # Write comparison report
    def row(name: str, r: dict[str, Any]) -> str:
        return (
            f"| {name} | {r['num_chunks']} | {r['avg_chunk_chars']:.0f} | {r['hit_at_k']:.3f} | {r['mrr_at_k']:.3f} |\n"
        )

    report = []
    report.append(f"## PART A — Chunking comparison (generated {datetime.now().isoformat(timespec='seconds')})\n")
    report.append("### Summary metrics\n")
    report.append("| Strategy | #chunks | Avg chars/chunk | Hit@5 | MRR@5 |\n")
    report.append("|---|---:|---:|---:|---:|\n")
    report.append(row("A: fixed_chars (1200, overlap 200)", res_a))
    report.append(row("B: fixed_words (260, overlap 50)", res_b))
    report.append(row("C: paragraph_packed (max 1400, overlap 1 para)", res_c))
    report.append("\n### Notes\n")
    report.append("- Retrieval uses the **same TF‑IDF + cosine similarity** pipeline for all strategies.\n")
    report.append("- Queries are mixed across the PDF and CSV to reflect the combined dataset.\n")
    report.append("- Ground truth uses keyword inclusion to keep the comparison consistent and reproducible.\n")
    report.append("\n")
    (OUT_DIR / "chunking_comparison.md").write_text("".join(report), encoding="utf-8")

    # Console output (high-level)
    print("Wrote outputs to:", OUT_DIR)
    print("Fixed chars:", {k: res_a[k] for k in ("num_chunks", "hit_at_k", "mrr_at_k")})
    print("Fixed words:", {k: res_b[k] for k in ("num_chunks", "hit_at_k", "mrr_at_k")})
    print("Paragraph packed:", {k: res_c[k] for k in ("num_chunks", "hit_at_k", "mrr_at_k")})


if __name__ == "__main__":
    main()
