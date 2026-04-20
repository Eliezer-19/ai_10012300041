"""
PART D — Full RAG pipeline: Query → Retrieval → Context selection → Prompt → LLM → Response.

Student: Eliezer Anim-Somuah · Index: 10012300041

Reuses Part B (retrieval), Part C (context packing, prompts, generators).
Interactive use: **`streamlit_app.py`** (no separate RAG CLI).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from part_c.context_pack import pack_context_for_prompt
from part_c.generation_backend import TextGenerator, generator_label, get_generator
from part_c.prompt_templates import PromptVariant, render_prompt

from part_b.cross_encoder_rerank import cross_encoder_rerank
from part_b.retrieval_system import HybridRetriever, load_bundle
from academic_city.constants import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_FEEDBACK_STORE,
    DEFAULT_INDEX_DIR,
    LOCKED_RERANK_CANDIDATE_K,
)
from academic_city.text_utils import preview_text
from part_g.feedback_loop import apply_feedback_rerank

LOG = logging.getLogger("academic_city.rag")


@dataclass
class RetrievedItem:
    rank: int
    chunk_id: str
    source: str
    doc_type: str
    score: float
    score_kind: str
    text_preview: str


@dataclass
class RAGResult:
    query: str
    prompt_variant: str
    llm_label: str
    retrieved: list[RetrievedItem]
    context_blocks_used: int
    context_chars: int
    context_pack_meta: dict[str, Any]
    final_prompt: str
    response: str
    stage_timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    """
    End-to-end RAG for Academic City. LLM is **Ollama only** via `get_generator` (Part C).
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: TextGenerator,
        *,
        prompt_variant: PromptVariant = PromptVariant.V2_GROUNDED,
        retrieve_k: int = 8,
        max_total_chars: int = 4500,
        max_chunk_chars: int = 2800,
        min_score: float | None = None,
        max_new_tokens: int = 512,
        feedback_store_path: Path | None = None,
        feedback_weight: float = 0.25,
        feedback_min_sim: float = 0.68,
        feedback_pool_mult: int = 3,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
        rerank_candidate_k: int = LOCKED_RERANK_CANDIDATE_K,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.prompt_variant = prompt_variant
        self.retrieve_k = retrieve_k
        self.max_total_chars = max_total_chars
        self.max_chunk_chars = max_chunk_chars
        self.min_score = min_score
        self.max_new_tokens = max_new_tokens
        self.feedback_store_path = feedback_store_path
        self.feedback_weight = feedback_weight
        self.feedback_min_sim = feedback_min_sim
        self.feedback_pool_mult = max(1, feedback_pool_mult)
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_model = cross_encoder_model
        self.rerank_candidate_k = max(1, rerank_candidate_k)
        self._llm_label = generator_label(generator)

    @classmethod
    def from_saved_index(
        cls,
        index_dir: Path | None = None,
        *,
        ollama_model: str | None = None,
        ollama_host: str | None = None,
        feedback_store_path: Path | None = None,
        use_feedback: bool = False,
        **kwargs: Any,
    ) -> RAGPipeline:
        index_dir = index_dir or DEFAULT_INDEX_DIR
        retriever = load_bundle(index_dir)
        gen = get_generator(
            "ollama",
            ollama_model=ollama_model,
            ollama_host=ollama_host,
        )
        extra = dict(kwargs)
        extra.pop("use_feedback", None)
        fb_path = feedback_store_path
        if use_feedback and fb_path is None:
            fb_path = DEFAULT_FEEDBACK_STORE
        if fb_path is not None:
            extra["feedback_store_path"] = fb_path
        return cls(retriever, gen, **extra)

    def run(self, query: str) -> RAGResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        LOG.info("stage=query | query=%r", query)

        # --- Retrieval (optional larger pool before feedback / cross-encoder rerank) ---
        t_r0 = time.perf_counter()
        if self.use_cross_encoder:
            base_pool = max(self.rerank_candidate_k, self.retrieve_k)
            if self.feedback_store_path and self.feedback_store_path.is_file():
                pool_k = max(base_pool, self.retrieve_k * self.feedback_pool_mult)
            else:
                pool_k = base_pool
        else:
            pool_k = self.retrieve_k
            if self.feedback_store_path and self.feedback_store_path.is_file():
                pool_k = max(self.retrieve_k * self.feedback_pool_mult, self.retrieve_k)

        hits = self.retriever.retrieve_hybrid_rrf(query, pool_k)
        if self.feedback_store_path and self.feedback_store_path.is_file():
            hits = apply_feedback_rerank(
                hits,
                query,
                self.retriever.embedder,
                self.feedback_store_path,
                min_sim=self.feedback_min_sim,
                weight=self.feedback_weight,
            )

        timings["retrieval_ms"] = (time.perf_counter() - t_r0) * 1000

        t_ce0 = time.perf_counter()
        if self.use_cross_encoder:
            hits = cross_encoder_rerank(
                query,
                hits,
                self.cross_encoder_model,
                top_k=self.retrieve_k,
            )
            timings["cross_encoder_rerank_ms"] = (time.perf_counter() - t_ce0) * 1000
            LOG.info(
                "stage=cross_encoder_rerank | model=%s ms=%.2f k_out=%s",
                self.cross_encoder_model,
                timings["cross_encoder_rerank_ms"],
                self.retrieve_k,
            )
        else:
            hits = hits[: self.retrieve_k]

        retrieved: list[RetrievedItem] = []
        for i, h in enumerate(hits, start=1):
            retrieved.append(
                RetrievedItem(
                    rank=i,
                    chunk_id=h.chunk_id,
                    source=h.source,
                    doc_type=h.doc_type,
                    score=h.score,
                    score_kind=h.score_kind,
                    text_preview=preview_text(h.text),
                )
            )
            LOG.info(
                "stage=retrieval | rank=%s chunk_id=%s score=%.6f score_kind=%s doc_type=%s source=%s",
                i,
                h.chunk_id,
                h.score,
                h.score_kind,
                h.doc_type,
                h.source,
            )

        # --- Context selection ---
        t_c0 = time.perf_counter()
        eff_max = self.max_total_chars
        eff_chunk = self.max_chunk_chars
        packed = pack_context_for_prompt(
            hits,
            max_total_chars=eff_max,
            max_chunk_chars=eff_chunk,
            min_score=self.min_score,
        )
        timings["context_pack_ms"] = (time.perf_counter() - t_c0) * 1000

        LOG.info(
            "stage=context | blocks_used=%s chars=%s dropped=%s meta=%s",
            packed.blocks_used,
            packed.chars_total,
            packed.dropped_count,
            packed.meta,
        )

        # --- Prompt ---
        t_p0 = time.perf_counter()
        prompt = render_prompt(
            self.prompt_variant,
            user_query=query,
            packed_context=packed.numbered_text,
        )
        timings["prompt_build_ms"] = (time.perf_counter() - t_p0) * 1000

        LOG.info("stage=prompt | variant=%s llm=%s prompt_chars=%s", self.prompt_variant.value, self._llm_label, len(prompt))
        LOG.debug("stage=prompt_full | content=%s", prompt)

        # --- LLM ---
        t_g0 = time.perf_counter()
        response = self.generator.generate(prompt, max_new_tokens=self.max_new_tokens)
        timings["generation_ms"] = (time.perf_counter() - t_g0) * 1000
        timings["total_ms"] = (time.perf_counter() - t0) * 1000

        LOG.info(
            "stage=llm | backend=%s response_chars=%s",
            self._llm_label,
            len(response),
        )
        LOG.info("stage=response | text=%r", response[:2000] + ("..." if len(response) > 2000 else ""))

        return RAGResult(
            query=query,
            prompt_variant=self.prompt_variant.value,
            llm_label=self._llm_label,
            retrieved=retrieved,
            context_blocks_used=packed.blocks_used,
            context_chars=packed.chars_total,
            context_pack_meta=dict(packed.meta),
            final_prompt=prompt,
            response=response,
            stage_timings_ms=timings,
        )
