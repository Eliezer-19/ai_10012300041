"""
PART C — Prompt template iterations for Academic City RAG.

Student: Eliezer Anim-Somuah · Index: 10012300041

Variants:
  - v1_minimal: inject context with little instruction (hallucination risk)
  - v2_grounded: strict grounding + abstention + citation
  - v3_json: same as v2 but forces structured output for easier validation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PromptVariant(str, Enum):
    V1_MINIMAL = "v1_minimal"
    V2_GROUNDED = "v2_grounded"
    V3_JSON = "v3_json"


@dataclass(frozen=True)
class PromptSpec:
    variant: PromptVariant
    name: str
    description: str


PROMPT_SPECS: dict[PromptVariant, PromptSpec] = {
    PromptVariant.V1_MINIMAL: PromptSpec(
        variant=PromptVariant.V1_MINIMAL,
        name="Minimal instruction",
        description="Context + question only; no explicit abstention or citation rules.",
    ),
    PromptVariant.V2_GROUNDED: PromptSpec(
        variant=PromptVariant.V2_GROUNDED,
        name="Grounded + abstain + cite",
        description="Use only passages; abstain if unsupported; cite [n] for facts.",
    ),
    PromptVariant.V3_JSON: PromptSpec(
        variant=PromptVariant.V3_JSON,
        name="Structured JSON",
        description="Same grounding as v2; output JSON for auditable abstain/citations.",
    ),
}


def render_prompt(
    variant: PromptVariant,
    *,
    user_query: str,
    packed_context: str,
) -> str:
    """Full prompt string for the generator (instruction-tuned models work best with a single block)."""

    if variant == PromptVariant.V1_MINIMAL:
        return f"""You are helping a student at Academic City University College.

Context from retrieved documents:
{packed_context}

Question: {user_query}

Answer concisely."""

    if variant == PromptVariant.V2_GROUNDED:
        return f"""You are an Academic City assistant. Answer using ONLY the numbered passages below.

Rules (hallucination control):
- Do not use outside knowledge; if the passages do not support an answer, reply exactly:
  "Not found in the provided context."
- For every factual claim, cite the passage number in square brackets, e.g. [1].
- If you combine passages, cite each, e.g. [1][2].

Passages:
{packed_context}

Question: {user_query}

Answer:"""

    if variant == PromptVariant.V3_JSON:
        return f"""You are an Academic City assistant. Use ONLY the numbered passages below.

Rules:
- Do not use outside knowledge (including geography or world facts not stated in the passages).
- Output a single JSON object with keys: "answer" (string), "citations" (array of integers = passage numbers), "abstain" (boolean).
- Set abstain to true if the passages do not contain enough information to answer; then "answer" is a short reason.
- If abstain is false, every factual claim must be supported by at least one citation integer.
- Output raw JSON only (no markdown fences).

Passages:
{packed_context}

Question: {user_query}

JSON:"""

    raise ValueError(f"Unknown variant: {variant}")
