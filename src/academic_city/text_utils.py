"""Shared text helpers (single-line previews, truncation)."""

from __future__ import annotations


def preview_text(text: str, max_chars: int = 320) -> str:
    """Collapse whitespace to one line and truncate with ``...`` for logs and reports."""
    t = text.replace("\n", " ").strip()
    return t if len(t) <= max_chars else t[: max_chars - 3] + "..."
