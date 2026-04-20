"""
Academic City RAG — Streamlit UI.

Student: Eliezer Anim-Somuah · Index: 10012300041

Run from repository root:
  pip install -r requirements-rag.txt
  pip install -e .
  python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Streamlit Cloud installs deps from requirements.txt only; without `pip install -e .`, packages
# under `src/` are not on PYTHONPATH. Support both editable installs and plain clones.
_APP_DIR = Path(__file__).resolve().parent
_SRC_DIR = _APP_DIR / "src"
if _SRC_DIR.is_dir():
    _src = str(_SRC_DIR)
    if _src not in sys.path:
        sys.path.insert(0, _src)

import streamlit as st

logging.basicConfig(level=logging.WARNING)
for _name in ("academic_city.rag", "sentence_transformers", "transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from academic_city.constants import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_MODEL_CLOUD,
    LOCKED_FEEDBACK_MIN_SIM,
    LOCKED_FEEDBACK_POOL_MULT,
    LOCKED_FEEDBACK_STORE,
    LOCKED_FEEDBACK_WEIGHT,
    LOCKED_INDEX_DIR,
    LOCKED_MAX_CONTEXT_CHARS,
    LOCKED_MAX_NEW_TOKENS,
    LOCKED_RERANK_CANDIDATE_K,
    LOCKED_USE_CROSS_ENCODER,
)
from part_c.prompt_templates import PROMPT_SPECS, PromptVariant
from part_d.rag_pipeline import RAGPipeline
from part_g.feedback_loop import FeedbackStore


def _is_streamlit_community_cloud_mount() -> bool:
    """True when running on Streamlit Community Cloud (repos are cloned under ``/mount/src/``)."""
    try:
        return "/mount/src/" in str(Path(__file__).resolve())
    except Exception:
        return False


def _get_ollama_api_key() -> str | None:
    """Ollama Cloud (ollama.com) uses ``OLLAMA_API_KEY`` as ``Authorization: Bearer``."""
    try:
        if st.secrets and "OLLAMA_API_KEY" in st.secrets:
            k = str(st.secrets["OLLAMA_API_KEY"]).strip()
            return k or None
    except Exception:
        pass
    k = (os.environ.get("OLLAMA_API_KEY") or "").strip()
    return k or None


def _default_ollama_host_input() -> str:
    """Prefill sidebar: Secrets → env → local-only default."""
    try:
        if st.secrets and "OLLAMA_HOST" in st.secrets:
            return str(st.secrets["OLLAMA_HOST"]).strip()
    except Exception:
        pass
    v = (os.environ.get("OLLAMA_HOST") or "").strip()
    if v:
        return v
    # Ollama Cloud: host is https://ollama.com — use when an API key exists and no host set.
    if _get_ollama_api_key():
        return "https://ollama.com"
    if _is_streamlit_community_cloud_mount():
        return ""
    return "http://127.0.0.1:11434"


def _resolve_ollama_host(sidebar_value: str) -> str:
    """Effective host: non-empty sidebar wins, else Secrets, else env, else localhost (local dev only)."""
    t = (sidebar_value or "").strip()
    if t:
        return t.rstrip("/")
    try:
        if st.secrets and "OLLAMA_HOST" in st.secrets:
            return str(st.secrets["OLLAMA_HOST"]).strip().rstrip("/")
    except Exception:
        pass
    e = (os.environ.get("OLLAMA_HOST") or "").strip()
    if e:
        return e.rstrip("/")
    if _get_ollama_api_key():
        return "https://ollama.com"
    if not _is_streamlit_community_cloud_mount():
        return "http://127.0.0.1:11434"
    return ""


def _is_loopback_ollama_url(url: str) -> bool:
    u = (url or "").lower()
    return "127.0.0.1" in u or "localhost" in u


def _default_ollama_model_input() -> str:
    try:
        if st.secrets and "OLLAMA_MODEL" in st.secrets:
            return str(st.secrets["OLLAMA_MODEL"]).strip()
    except Exception:
        pass
    env = (os.environ.get("OLLAMA_MODEL") or "").strip()
    if env:
        return env
    # ollama.com uses different tags than local Ollama; llama3 often is not valid on the cloud API.
    if _get_ollama_api_key():
        return DEFAULT_OLLAMA_MODEL_CLOUD
    return DEFAULT_OLLAMA_MODEL


def _feedback_mtime(path: Path | None) -> float:
    if path is not None and path.is_file():
        return path.stat().st_mtime
    return 0.0


@st.cache_resource
def build_pipeline(
    locked_index_str: str,
    ollama_model: str,
    ollama_host: str,
    ollama_api_key: str | None,
    retrieve_k: int,
    use_feedback: bool,
    use_cross_encoder: bool,
    cross_encoder_model: str,
    prompt_variant_value: str,
    _feedback_mtime_key: float,
) -> RAGPipeline:
    index_dir = Path(locked_index_str) if locked_index_str.strip() else None
    return RAGPipeline.from_saved_index(
        index_dir,
        ollama_model=ollama_model,
        ollama_host=ollama_host or None,
        ollama_api_key=ollama_api_key,
        prompt_variant=PromptVariant(prompt_variant_value),
        retrieve_k=retrieve_k,
        max_total_chars=LOCKED_MAX_CONTEXT_CHARS,
        max_new_tokens=LOCKED_MAX_NEW_TOKENS,
        use_feedback=use_feedback,
        feedback_store_path=LOCKED_FEEDBACK_STORE if use_feedback else None,
        feedback_weight=LOCKED_FEEDBACK_WEIGHT,
        feedback_min_sim=LOCKED_FEEDBACK_MIN_SIM,
        feedback_pool_mult=LOCKED_FEEDBACK_POOL_MULT,
        use_cross_encoder=use_cross_encoder,
        cross_encoder_model=cross_encoder_model,
        rerank_candidate_k=LOCKED_RERANK_CANDIDATE_K,
    )


def main() -> None:
    st.set_page_config(
        page_title="Academic City RAG",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
<style>
  /* Subtle app-wide typography + spacing */
  .block-container { padding-top: 2.0rem; padding-bottom: 3.0rem; }
  .ac-muted { color: rgba(250,250,250,0.65); font-size: 0.95rem; }
  .ac-small { color: rgba(250,250,250,0.55); font-size: 0.85rem; }
  .ac-kpi { border: 1px solid rgba(250,250,250,0.10); border-radius: 12px; padding: 12px 14px; background: rgba(250,250,250,0.03); }
  .ac-pill { display:inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(250,250,250,0.12); font-size: 0.80rem; color: rgba(250,250,250,0.75); }
</style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        st.title("Academic City RAG")
        st.markdown(
            '<div class="ac-muted"><b>Eliezer Anim‑Somuah</b> · Index <b>10012300041</b></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ac-small">Corpus: Ghana 2025 budget PDF + Ghana election results CSV · Backend: Ollama</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
<div class="ac-kpi">
  <div class="ac-small">This UI displays (Part D requirements)</div>
  <div style="margin-top:6px">
    <span class="ac-pill">Retrieved docs</span>
    <span class="ac-pill">Similarity scores</span>
    <span class="ac-pill">Final prompt</span>
    <span class="ac-pill">Timings</span>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.header("Model")
        ollama_host = st.text_input(
            "Ollama host (URL)",
            value=_default_ollama_host_input(),
            placeholder="https://ollama.com or your self-hosted Ollama",
            help=(
                "**Self-hosted:** base URL for `/api/chat` (e.g. `https://your-vm:11434`). "
                "**Ollama Cloud (ollama.com):** use `https://ollama.com` and set **OLLAMA_API_KEY** in Secrets."
            ),
        )
        ollama_model = st.text_input(
            "Model name",
            value=_default_ollama_model_input(),
            help=(
                "**Self-hosted:** name from `ollama list` (e.g. llama3). "
                "**ollama.com:** use a model your API key supports — set **OLLAMA_MODEL** in Secrets if the default is wrong."
            ),
        )
        st.header("Retrieval + prompt")
        prompt_variant = st.selectbox(
            "Prompt variant",
            options=list(PromptVariant),
            index=list(PromptVariant).index(PromptVariant.V2_GROUNDED),
            format_func=lambda v: f"{PROMPT_SPECS[v].name} (`{v.value}`)",
            help="Controls grounding style (default: v2_grounded).",
        )
        st.caption(PROMPT_SPECS[prompt_variant].description)
        retrieve_k = st.slider(
            "retrieve_k",
            1,
            24,
            8,
            help="How many passages are kept after hybrid retrieval, feedback (if on), and cross-encoder rerank (if on); those passages are packed into the prompt.",
        )
        use_cross_encoder = st.checkbox(
            "Cross-encoder rerank",
            value=LOCKED_USE_CROSS_ENCODER,
            help=(
                "Re-score the top hybrid candidates with a small MS MARCO cross-encoder before packing. "
                "Often improves ordering vs. only increasing retrieve_k."
            ),
        )

        st.header("Feedback (Part G)")
        use_feedback = st.checkbox("Use feedback-augmented retrieval", value=False)

        st.divider()
        if st.button("Reload pipeline (clear cache)"):
            st.cache_resource.clear()
            st.session_state.pop("last_result", None)
            st.rerun()

    mkey = _feedback_mtime(LOCKED_FEEDBACK_STORE) if use_feedback else 0.0
    cross_encoder_model = os.environ.get("CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL)

    ollama_resolved = _resolve_ollama_host(ollama_host)
    ollama_api_key = _get_ollama_api_key()
    cloud_ok = bool(ollama_api_key) and "ollama.com" in ollama_resolved.lower()
    if _is_streamlit_community_cloud_mount() and (
        not ollama_resolved or _is_loopback_ollama_url(ollama_resolved)
    ) and not cloud_ok:
        st.error(
            "Ollama is not reachable at **127.0.0.1** on Streamlit Cloud. "
            "Use **Ollama Cloud** (ollama.com + API key) or a **public / reachable** self-hosted Ollama URL."
        )
        st.markdown(
            "**Option A — Ollama Cloud:** App → **Secrets**:\n\n"
            "```toml\n"
            'OLLAMA_API_KEY = "your_key_here"\n'
            'OLLAMA_HOST = "https://ollama.com"\n'
            'OLLAMA_MODEL = "gpt-oss:120b-cloud"\n'
            "```\n\n"
            "**Option B — Self-hosted:** expose Ollama (ideally HTTPS + auth) and set `OLLAMA_HOST` to that base URL.\n\n"
            "Do not use `/api/chat` or `/api/generate` in the host field — base URL only."
        )
        st.stop()

    try:
        pipeline = build_pipeline(
            str(LOCKED_INDEX_DIR),
            ollama_model,
            ollama_resolved,
            ollama_api_key,
            retrieve_k,
            use_feedback,
            use_cross_encoder,
            cross_encoder_model,
            prompt_variant.value,
            mkey,
        )
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        st.info(
            "If deploying on Streamlit Cloud: host Ollama on a VM and set `OLLAMA_HOST` in Streamlit Secrets, "
            "or paste the URL in the sidebar."
        )
        st.stop()

    q_col, run_col = st.columns([6, 1], vertical_alignment="bottom")
    with q_col:
        query = st.text_area("Question", height=110, placeholder="Ask about the 2025 budget or Ghana election data…")
    with run_col:
        run = st.button("Run", type="primary", use_container_width=True)

    if run and query.strip():
        with st.spinner("Retrieving and generating…"):
            try:
                result = pipeline.run(query.strip())
            except Exception as e:
                st.exception(e)
                st.stop()
        st.session_state["last_query"] = query.strip()
        st.session_state["last_result"] = result

    res = st.session_state.get("last_result")
    if res is not None:
        # Top summary strip
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prompt", res.prompt_variant)
        m2.metric("LLM", res.llm_label.split("@")[0])
        m3.metric("Context blocks", res.context_blocks_used)
        m4.metric("Context chars", res.context_chars)

        tab_answer, tab_evidence, tab_prompt, tab_feedback = st.tabs(
            ["Answer", "Evidence", "Prompt", "Feedback"]
        )

        with tab_answer:
            st.subheader("Answer")
            st.markdown(res.response if res.response.strip() else "_(empty response)_")

        with tab_evidence:
            st.subheader("Retrieved passages (top-k)")
            st.caption("Each row shows rank, similarity score, doc_type/source, and the chunk id. Expand a row to see the preview.")

            rows = []
            for r in res.retrieved:
                rows.append(
                    {
                        "rank": r.rank,
                        "score": round(float(r.score), 6),
                        "score_kind": r.score_kind,
                        "doc_type": r.doc_type,
                        "source": r.source,
                        "chunk_id": r.chunk_id,
                    }
                )
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
                for r in res.retrieved:
                    with st.expander(f"[{r.rank}] score={r.score:.4f} · {r.doc_type} · {r.source}", expanded=(r.rank <= 2)):
                        st.code(r.chunk_id, language="text")
                        st.text(r.text_preview)
            else:
                st.info("No retrieved passages.")

            st.divider()
            st.subheader("Timings (ms)")
            st.json(res.stage_timings_ms)

            st.divider()
            st.subheader("Downloads")
            st.download_button(
                "Download result JSON",
                data=json.dumps(
                    {
                        "query": res.query,
                        "prompt_variant": res.prompt_variant,
                        "llm_label": res.llm_label,
                        "retrieved": [r.__dict__ for r in res.retrieved],
                        "context_blocks_used": res.context_blocks_used,
                        "context_chars": res.context_chars,
                        "context_pack_meta": res.context_pack_meta,
                        "final_prompt": res.final_prompt,
                        "response": res.response,
                        "stage_timings_ms": res.stage_timings_ms,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                file_name="rag_result.json",
                mime="application/json",
                use_container_width=True,
            )

        with tab_prompt:
            st.subheader("Final prompt sent to the model")
            st.code(res.final_prompt, language="text")
            st.download_button(
                "Download prompt (.txt)",
                data=res.final_prompt,
                file_name="final_prompt.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with tab_feedback:
            st.subheader("Record feedback (Part G)")
            st.caption(
                "Label a retrieved chunk to improve future retrieval for similar questions. "
                "Turn on **Use feedback-augmented retrieval** in the sidebar to apply it."
            )
            if not res.retrieved:
                st.info("No retrieved passages to label.")
            else:
                opts = [
                    f"[{r.rank}] {r.chunk_id[:80]}…" if len(r.chunk_id) > 80 else f"[{r.rank}] {r.chunk_id}"
                    for r in res.retrieved
                ]
                choice = st.selectbox("Chunk", range(len(res.retrieved)), format_func=lambda i: opts[i])
                last_q = st.session_state.get("last_query", "")
                store_path = LOCKED_FEEDBACK_STORE
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Helpful (+1)", key="fb_pos", type="primary", use_container_width=True):
                        ch = res.retrieved[choice].chunk_id
                        FeedbackStore(store_path).append(last_q, ch, 1)
                        st.success("Saved +1. Run the same question again to see the re-ranking effect.")
                with col_b:
                    if st.button("Not helpful (−1)", key="fb_neg", use_container_width=True):
                        ch = res.retrieved[choice].chunk_id
                        FeedbackStore(store_path).append(last_q, ch, -1)
                        st.success("Saved −1. Run the same question again to see the re-ranking effect.")


if __name__ == "__main__":
    main()
