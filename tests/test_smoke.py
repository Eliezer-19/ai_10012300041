"""Layout / import smoke tests — run after `pip install -e .` from repository root."""

from __future__ import annotations

from part_b.retrieval_system import RetrievedChunk
from part_c.context_pack import pack_context_for_prompt
from part_c.prompt_templates import PromptVariant, render_prompt
from part_d.rag_pipeline import RAGPipeline


def test_import_rag_pipeline() -> None:
    assert RAGPipeline is not None


def test_pack_context_and_render_prompt_shape() -> None:
    chunk = RetrievedChunk(
        chunk_id="test:1",
        source="test.pdf",
        doc_type="budget_pdf",
        text="Sample passage about VAT.",
        meta={},
        score=0.9,
        score_kind="hybrid",
    )
    packed = pack_context_for_prompt([chunk], max_total_chars=2000, max_chunk_chars=500)
    assert packed.blocks_used >= 1
    assert "[1]" in packed.numbered_text

    prompt = render_prompt(
        PromptVariant.V2_GROUNDED,
        user_query="What about VAT?",
        packed_context=packed.numbered_text,
    )
    assert "What about VAT?" in prompt
    assert packed.numbered_text in prompt
