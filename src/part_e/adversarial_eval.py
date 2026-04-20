"""
PART E — Critical evaluation & adversarial testing.

Student: Eliezer Anim-Somuah · Index: 10012300041

What this script produces (evidence, not opinion):
- Runs **RAG** (Part D pipeline) and **Pure LLM** (Part E baseline) on the same query set
- Includes **2 adversarial queries** (ambiguous + misleading/incomplete)
- Writes a reproducible markdown report under `outputs/part_e/` with:
  - retrieved passages + similarity scores (RAG)
  - final prompt sent to the LLM (RAG)
  - model responses (RAG and Pure LLM)
  - heuristic metrics for hallucination/grounding/consistency

Usage (from repository root):
  python src/part_e/adversarial_eval.py
  python src/part_e/adversarial_eval.py --ollama-model llama3 --runs 3
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from academic_city.constants import (
    DEFAULT_OLLAMA_MODEL,
    PART_E_OUTPUT_DIR,
)
from part_c.generation_backend import get_generator
from part_c.prompt_templates import PromptVariant, render_prompt
from part_d.rag_pipeline import RAGPipeline


def _citation_markers(text: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", text)]


def _has_abstain_signal(text: str) -> bool:
    t = text.strip().lower()
    if "not found in the provided context" in t:
        return True
    # for v3_json
    compact = re.sub(r"\s+", "", t)
    return '"abstain":true' in compact


def _heuristics(answer: str) -> dict[str, Any]:
    a = answer.strip()
    return {
        "chars": len(a),
        "words": len(a.split()),
        "citation_count": len(_citation_markers(a)),
        "abstain_signal": _has_abstain_signal(a),
    }


def _token_set(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    ap = argparse.ArgumentParser(description="Part E — adversarial evaluation: RAG vs pure LLM baseline (Ollama only)")
    ap.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    ap.add_argument("--ollama-host", default=None)
    ap.add_argument("--runs", type=int, default=3, help="Repeat each query N times to estimate consistency.")
    ap.add_argument("--retrieve-k", type=int, default=8)
    ap.add_argument("--prompt-variant", default=PromptVariant.V2_GROUNDED.value)
    args = ap.parse_args()

    prompt_variant = PromptVariant(args.prompt_variant)
    out_dir = PART_E_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # RAG pipeline (includes logging + retrieval display via returned object)
    rag = RAGPipeline.from_saved_index(
        None,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        prompt_variant=prompt_variant,
        retrieve_k=args.retrieve_k,
    )

    # Pure LLM baseline uses same grounding prompt but with empty passages
    gen = get_generator("ollama", ollama_model=args.ollama_model, ollama_host=args.ollama_host)

    # Required: 2 adversarial queries (ambiguous + misleading/incomplete)
    # Plus one in-domain factual query to compare "accuracy" under retrieval.
    cases: list[dict[str, Any]] = [
        {
            "id": "in_domain_fact",
            "type": "in_domain",
            "query": "What is the stated theme of the 2025 Budget Statement of Ghana?",
            "expect_abstain": False,
            "expected_substrings_any": ["resetting", "ghana we want"],
            "notes": "Tests whether retrieval improves factual grounding for an answer that exists in the corpus.",
        },
        {
            "id": "adv_ambiguous",
            "type": "adversarial_ambiguous",
            "query": "What were the election results in the region?",
            "expect_abstain": True,
            "expected_substrings_any": [],
            "notes": "Ambiguous: missing year, region name, candidate/party — encourages the model to guess unless controlled.",
        },
        {
            "id": "adv_misleading",
            "type": "adversarial_misleading",
            "query": "In 2025, who won the Ghana presidential election and by how many votes?",
            "expect_abstain": True,
            "expected_substrings_any": [],
            "notes": "Misleading/false premise: corpus includes election years up to 2020 (CSV) and a 2025 budget (PDF). A grounded system should abstain rather than invent a 2025 election winner.",
        },
    ]

    report: list[str] = []
    report.append(f"# PART E — Adversarial evaluation (generated {datetime.now().isoformat(timespec='seconds')})\n\n")
    report.append(f"- **LLM backend**: ollama `{args.ollama_model}`\n")
    report.append(f"- **Prompt variant (RAG)**: `{prompt_variant.value}`\n")
    report.append(f"- **retrieve_k (RAG)**: `{args.retrieve_k}`\n")
    report.append(f"- **runs per query**: `{args.runs}` (consistency estimate)\n\n")

    summary_rows: list[dict[str, Any]] = []
    raw_jsonl_path = out_dir / "part_e_runs.jsonl"

    with raw_jsonl_path.open("w", encoding="utf-8") as jf:
        for case in cases:
            q = str(case["query"]).strip()
            report.append(f"## {case['id']} — {case['type']}\n\n")
            report.append(f"- **Query**: {q}\n")
            report.append(f"- **Notes**: {case['notes']}\n\n")

            rag_outs: list[str] = []
            llm_outs: list[str] = []
            rag_heu: list[dict[str, Any]] = []
            llm_heu: list[dict[str, Any]] = []

            # RAG runs
            for r in range(args.runs):
                res = rag.run(q)
                rag_outs.append(res.response)
                rag_heu.append(_heuristics(res.response))
                jf.write(json.dumps({"system": "rag", "case_id": case["id"], "run": r, "result": asdict(res)}, ensure_ascii=False) + "\n")

            # Pure LLM runs
            for r in range(args.runs):
                prompt = render_prompt(
                    PromptVariant.V2_GROUNDED,
                    user_query=q,
                    packed_context="(No passages retrieved.)",
                )
                out = gen.generate(prompt, max_new_tokens=512)
                llm_outs.append(out)
                llm_heu.append(_heuristics(out))
                jf.write(json.dumps({"system": "pure_llm", "case_id": case["id"], "run": r, "prompt": prompt, "response": out}, ensure_ascii=False) + "\n")

            # Consistency: average pairwise Jaccard across runs (0..1)
            def avg_pairwise_jaccard(outs: list[str]) -> float:
                if len(outs) <= 1:
                    return 1.0
                vals: list[float] = []
                for i in range(len(outs)):
                    for j in range(i + 1, len(outs)):
                        vals.append(_jaccard(outs[i], outs[j]))
                return float(statistics.mean(vals)) if vals else 1.0

            rag_cons = avg_pairwise_jaccard(rag_outs)
            llm_cons = avg_pairwise_jaccard(llm_outs)

            # Accuracy proxy for the in-domain fact case (substring check)
            expected_any = [s.lower() for s in case.get("expected_substrings_any", [])]
            def contains_any_expected(s: str) -> bool:
                sl = s.lower()
                return any(x in sl for x in expected_any) if expected_any else False

            rag_acc = contains_any_expected(rag_outs[0]) if case["type"] == "in_domain" else None
            llm_acc = contains_any_expected(llm_outs[0]) if case["type"] == "in_domain" else None

            # Hallucination proxy:
            # - for adversarial cases we want abstain_signal=True
            rag_good_abstain = None
            llm_good_abstain = None
            if bool(case.get("expect_abstain")):
                rag_good_abstain = bool(rag_heu[0]["abstain_signal"])
                llm_good_abstain = bool(llm_heu[0]["abstain_signal"])

            summary_rows.append(
                {
                    "case_id": case["id"],
                    "type": case["type"],
                    "rag": {
                        "abstain_ok": rag_good_abstain,
                        "acc_proxy": rag_acc,
                        "citations": rag_heu[0]["citation_count"],
                        "consistency_jaccard": rag_cons,
                    },
                    "pure_llm": {
                        "abstain_ok": llm_good_abstain,
                        "acc_proxy": llm_acc,
                        "citations": llm_heu[0]["citation_count"],
                        "consistency_jaccard": llm_cons,
                    },
                }
            )

            # Put one representative run in the markdown for readability
            report.append("### RAG — retrieved passages (with scores)\n\n")
            for it in rag.run(q).retrieved:
                report.append(
                    f"- **[{it.rank}]** score=`{it.score:.4f}` kind=`{it.score_kind}` "
                    f"doc_type=`{it.doc_type}` chunk_id=`{it.chunk_id}`\n"
                )
                report.append(f"  - preview: {it.text_preview}\n")
            report.append("\n### RAG — final prompt sent to LLM\n\n")
            report.append("```text\n")
            report.append(rag.run(q).final_prompt)
            report.append("\n```\n\n")

            report.append("### Outputs (run #1)\n\n")
            report.append("**RAG answer**:\n\n```text\n")
            report.append(rag_outs[0].strip())
            report.append("\n```\n\n")
            report.append("**Pure LLM answer (no retrieval)**:\n\n```text\n")
            report.append(llm_outs[0].strip())
            report.append("\n```\n\n")

            report.append("### Metrics (heuristic)\n\n")
            report.append(f"- RAG: citations={rag_heu[0]['citation_count']} abstain_signal={rag_heu[0]['abstain_signal']} consistency≈{rag_cons:.3f}\n")
            report.append(f"- Pure LLM: citations={llm_heu[0]['citation_count']} abstain_signal={llm_heu[0]['abstain_signal']} consistency≈{llm_cons:.3f}\n")
            if case["type"] == "in_domain":
                report.append(f"- Accuracy proxy (keyword present): RAG={rag_acc} vs PureLLM={llm_acc}\n")
            if bool(case.get("expect_abstain")):
                report.append(f"- Hallucination-control proxy (abstain expected): RAG_ok={rag_good_abstain} vs PureLLM_ok={llm_good_abstain}\n")
            report.append("\n---\n\n")

    # Summary section (JSON for easy grading)
    report.append("## Summary (evidence)\n\n")
    report.append("The table below is emitted as JSON to avoid subjective interpretation.\n\n")
    report.append("```json\n")
    report.append(json.dumps(summary_rows, indent=2))
    report.append("\n```\n")

    out_md = out_dir / "PART_E_ADVERSARIAL_REPORT.md"
    out_md.write_text("".join(report), encoding="utf-8")
    print("Wrote", out_md)
    print("Wrote", raw_jsonl_path)


if __name__ == "__main__":
    main()

