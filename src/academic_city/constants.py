"""Single source of truth for default paths and Streamlit-locked RAG budgets."""

from __future__ import annotations

from pathlib import Path

from academic_city.paths import project_root

_ROOT = project_root()

# All generated artifacts live under one tree: <repo>/outputs/<part_*>/
OUTPUTS_ROOT: Path = _ROOT / "outputs"
PART_A_OUTPUT_DIR: Path = OUTPUTS_ROOT / "part_a"
PART_B_OUTPUT_DIR: Path = OUTPUTS_ROOT / "part_b"
PART_C_OUTPUT_DIR: Path = OUTPUTS_ROOT / "part_c"
PART_E_OUTPUT_DIR: Path = OUTPUTS_ROOT / "part_e"
PART_G_OUTPUT_DIR: Path = OUTPUTS_ROOT / "part_g"

# Default data files (Part B reads chunks produced by Part A)
DEFAULT_CHUNKS_JSONL: Path = PART_A_OUTPUT_DIR / "chunks_paragraph_packed.jsonl"

# Index & feedback
DEFAULT_INDEX_DIR: Path = PART_B_OUTPUT_DIR
DEFAULT_FEEDBACK_STORE: Path = PART_G_OUTPUT_DIR / "feedback.jsonl"

# Streamlit / shared defaults (also documented in DOCUMENTATION.md)
LOCKED_INDEX_DIR: Path = DEFAULT_INDEX_DIR
LOCKED_MAX_CONTEXT_CHARS: int = 4500
LOCKED_MAX_NEW_TOKENS: int = 512
LOCKED_FEEDBACK_STORE: Path = DEFAULT_FEEDBACK_STORE
LOCKED_FEEDBACK_WEIGHT: float = 0.25
LOCKED_FEEDBACK_MIN_SIM: float = 0.68
LOCKED_FEEDBACK_POOL_MULT: int = 3
DEFAULT_OLLAMA_MODEL: str = "llama3"

# Cross-encoder rerank (hybrid top-k → CE scores → final top retrieve_k)
DEFAULT_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCKED_USE_CROSS_ENCODER: bool = True
LOCKED_RERANK_CANDIDATE_K: int = 24
