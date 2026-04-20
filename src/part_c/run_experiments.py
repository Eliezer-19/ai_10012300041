"""
PART C — Run same queries under different prompt templates; record outputs + simple metrics.

Student: Eliezer Anim-Somuah · Index: 10012300041

LLM: **Ollama only** (same as Streamlit / Part D).

Usage (from repository root):
  python src/part_c/run_experiments.py
  python src/part_c/run_experiments.py --ollama-model llama3
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

from academic_city.constants import DEFAULT_INDEX_DIR, DEFAULT_OLLAMA_MODEL, PART_C_OUTPUT_DIR
from part_c.context_pack import pack_context_for_prompt
from part_c.generation_backend import OllamaChatGenerator, get_generator
from part_c.prompt_templates import PROMPT_SPECS, PromptVariant, render_prompt

from part_b.retrieval_system import load_bundle


PART_B_OUT = DEFAULT_INDEX_DIR


def analyze_output(text: str, *, expect_abstain: bool) -> dict:
    """Lightweight heuristics for experiment evidence (not a full evaluator)."""
    t = text.strip()
    cites = len(re.findall(r"\[\d+\]", t))
    tl = t.lower()
    abstain_hit = (
        "not found in the provided context" in tl
        or '"abstain": true' in t.replace(" ", "").lower()
        or '"abstain":true' in t.replace(" ", "").lower()
        or "abstain\": true" in t.replace(" ", "").lower()
    )
    return {
        "chars": len(t),
        "words": len(t.split()),
        "citation_markers": cites,
        "abstain_signal": abstain_hit,
        "likely_good_for_ood": abstain_hit if expect_abstain else None,
    }


def run(
    *,
    ollama_model: str | None,
    ollama_host: str | None,
    retrieve_k: int,
    max_context_chars: int,
) -> None:
    out_dir = PART_C_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    retriever = load_bundle(PART_B_OUT)
    gen = get_generator("ollama", ollama_model=ollama_model, ollama_host=ollama_host)
    assert isinstance(gen, OllamaChatGenerator)
    gen_label = f"{type(gen).__name__} ({gen.model} @ {gen.base_url})"

    experiments: list[dict] = [
        {
            "id": "exp1",
            "query": "What is the stated theme of the 2025 Budget Statement of Ghana?",
            "expect_abstain": False,
            "ground_truth_substrings": ["resetting", "ghana we want"],
        },
        {
            "id": "exp2",
            "query": "What is the capital city of France?",
            "expect_abstain": True,
            "ground_truth_substrings": [],
        },
    ]

    variants = [PromptVariant.V1_MINIMAL, PromptVariant.V2_GROUNDED, PromptVariant.V3_JSON]

    lines: list[str] = []
    lines.append(f"# PART C — Prompt experiments ({datetime.now().isoformat(timespec='seconds')})\n\n")
    lines.append(f"- **Generator**: {gen_label}\n")
    lines.append(
        f"- **Retrieval**: hybrid RRF from `outputs/part_b`, top-{retrieve_k}, packed ≤{max_context_chars} chars.\n\n"
    )

    for exp in experiments:
        lines.append(f"## {exp['id']}: {exp['query']}\n\n")
        hits = retriever.retrieve_hybrid_rrf(exp["query"], retrieve_k)
        packed = pack_context_for_prompt(
            hits,
            max_total_chars=max_context_chars,
            max_chunk_chars=2800,
            min_score=None,
        )
        lines.append(f"- **Context blocks used**: {packed.blocks_used} ({packed.chars_total} chars)\n\n")

        for pv in variants:
            spec = PROMPT_SPECS[pv]
            prompt = render_prompt(pv, user_query=exp["query"], packed_context=packed.numbered_text)
            max_tokens = 400
            out = gen.generate(prompt, max_new_tokens=max_tokens)
            metrics = analyze_output(out, expect_abstain=exp["expect_abstain"])
            grounded = False
            if not exp["expect_abstain"]:
                low = out.lower()
                grounded = any(s in low for s in exp["ground_truth_substrings"])

            lines.append(f"### {pv.value} — {spec.name}\n\n")
            lines.append(f"*{spec.description}*\n\n")
            lines.append("**Metrics (heuristic)**:\n")
            lines.append(f"- words={metrics['words']}, citation_markers={metrics['citation_markers']}, ")
            lines.append(f"abstain_signal={metrics['abstain_signal']}\n")
            if not exp["expect_abstain"]:
                lines.append(f"- contains_ground_truth_keyword: **{grounded}** (theme wording)\n")
            lines.append("\n**Model output**:\n\n```\n")
            lines.append(out)
            lines.append("\n```\n\n")

        lines.append("---\n\n")

    lines.append("## Analysis — observed differences (see also `DOCUMENTATION.md` in project root)\n\n")
    lines.append(
        "- **v1_minimal**: minimal guardrails; on the OOD geography probe the model may answer from **prior knowledge** "
        "(e.g. “Paris”) even when retrieved passages are unrelated.\n"
    )
    lines.append(
        "- **v2_grounded**: explicit **abstention** phrase and **context-only** rules often fix this: the model outputs "
        "the required **Not found in the provided context.** when passages do not support the answer.\n"
    )
    lines.append(
        "- **v3_json**: JSON is useful for UI/validators, but smaller Ollama models may still **hallucinate structured fields** "
        "(e.g. `abstain: false` with invented citations). Treat JSON as **best-effort** unless you add schema validation "
        "or use a larger / stronger Ollama model.\n"
    )

    out_md = out_dir / "PART_C_EXPERIMENT_RESULTS.md"
    out_md.write_text("".join(lines), encoding="utf-8")
    print("Wrote", out_md)

    _packed = pack_context_for_prompt(
        retriever.retrieve_hybrid_rrf(experiments[0]["query"], retrieve_k),
        max_total_chars=max_context_chars,
    )
    sample = render_prompt(
        PromptVariant.V2_GROUNDED,
        user_query=experiments[0]["query"],
        packed_context=_packed.numbered_text,
    )
    (out_dir / "sample_prompt_v2_grounded.txt").write_text(sample, encoding="utf-8")
    print("Wrote", out_dir / "sample_prompt_v2_grounded.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description="Part C prompt experiments (Ollama only)")
    ap.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        help="Ollama model tag (env OLLAMA_MODEL)",
    )
    ap.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST"),
        help="Ollama API base (env OLLAMA_HOST)",
    )
    ap.add_argument("--retrieve-k", type=int, default=8)
    ap.add_argument("--max-context-chars", type=int, default=4500)
    args = ap.parse_args()
    run(
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host or None,
        retrieve_k=args.retrieve_k,
        max_context_chars=args.max_context_chars,
    )


if __name__ == "__main__":
    main()
