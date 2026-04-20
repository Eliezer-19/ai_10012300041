"""Project root resolution."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repository root (contains `src/`, `outputs/`, data files)."""
    # This file: <root>/src/academic_city/paths.py
    return Path(__file__).resolve().parent.parent.parent
