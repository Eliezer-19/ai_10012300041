"""
Part E — Pure LLM baseline (no retrieval): same v2_grounded rules, empty passages.

Student: Eliezer Anim-Somuah · Index: 10012300041

LLM: **Ollama only**.

Usage (from repository root):
  python src/part_e/pure_llm_baseline.py "Your question"
  python src/part_e/pure_llm_baseline.py --ollama-model llama3 "Who won the presidential election?"

Compare output to RAG with retrieval: run `python -m streamlit run streamlit_app.py` with the same question (use **`v2_grounded`** in the UI).
"""

from __future__ import annotations

import argparse
import os

from academic_city.constants import DEFAULT_OLLAMA_MODEL, LOCKED_MAX_NEW_TOKENS
from part_c.generation_backend import get_generator
from part_c.prompt_templates import PromptVariant, render_prompt


def main() -> None:
    ap = argparse.ArgumentParser(description="Pure LLM baseline — v2_grounded with empty context (Part E)")
    ap.add_argument("query", help="User question")
    ap.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    ap.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST"))
    ap.add_argument("--max-new-tokens", type=int, default=LOCKED_MAX_NEW_TOKENS)
    args = ap.parse_args()

    gen = get_generator(
        "ollama",
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host or None,
    )
    prompt = render_prompt(
        PromptVariant.V2_GROUNDED,
        user_query=args.query.strip(),
        packed_context="(No passages retrieved.)",
    )
    out = gen.generate(prompt, max_new_tokens=args.max_new_tokens)
    print("=== Pure LLM (no retrieval) — v2_grounded, empty passages ===\n")
    print(out)


if __name__ == "__main__":
    main()
