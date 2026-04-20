"""
Text generation for Part C / Part D — **Ollama only** (local HTTP `/api/chat`).

Student: Eliezer Anim-Somuah · Index: 10012300041

Requires `ollama serve` and a pulled model (e.g. `ollama pull llama3`).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from academic_city.constants import DEFAULT_OLLAMA_MODEL


class TextGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        ...


class OllamaChatGenerator(TextGenerator):

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": max_new_tokens, "temperature": 0.2},
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace").strip()
                if err_body:
                    detail = f" Body: {err_body[:1200]}{'…' if len(err_body) > 1200 else ''}"
            except Exception:
                pass

            if e.code == 404:
                hint = (
                    f"Model missing or wrong name (404). Run `ollama list` and "
                    f"`ollama pull {self.model}` — names must match exactly."
                )
            elif e.code == 500:
                hint = (
                    "Internal server error — common causes: prompt/context too large for the model, "
                    "GPU or system RAM exhausted, or the runner crashed. "
                    "In the app: open the sidebar and lower **retrieve_k** (or turn off cross-encoder "
                    "only if you suspect load; usually shrinking context helps). "
                    "Then try `ollama ps`, restart Ollama, or a smaller/fewer-layers model. "
                    f"Confirm the model with `ollama list` and `ollama pull {self.model}` if needed."
                )
            else:
                hint = (
                    f"Check Ollama is healthy (`ollama list`), pull if needed: "
                    f"`ollama pull {self.model}`."
                )

            raise RuntimeError(
                f"Ollama HTTP {e.code} at {self.base_url}/api/chat — {e.reason}.{detail} {hint}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Is `ollama serve` running? ({e})"
            ) from e

        msg = raw.get("message") or {}
        return (msg.get("content") or "").strip()


def get_generator(
    prefer: str = "ollama",
    *,
    ollama_model: str | None = None,
    ollama_host: str | None = None,
) -> OllamaChatGenerator:
    """
    Return the Ollama chat generator. Only ``\"ollama\"`` and ``\"auto\"`` are accepted (both use Ollama).
    """
    if prefer not in ("ollama", "auto"):
        raise ValueError(f"Only Ollama is supported; got prefer={prefer!r}")
    return OllamaChatGenerator(model=ollama_model, base_url=ollama_host)


def generator_label(gen: TextGenerator) -> str:
    """Human-readable backend name for logs and UI."""
    if isinstance(gen, OllamaChatGenerator):
        return f"ollama:{gen.model}@{gen.base_url}"
    return type(gen).__name__
