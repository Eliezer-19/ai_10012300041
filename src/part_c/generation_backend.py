"""
Text generation for Part C / Part D — **Ollama** (local `/api/chat` or **ollama.com** `/api/generate`).

Student: Eliezer Anim-Somuah · Index: 10012300041

- **Local**: ``OLLAMA_HOST`` (default ``http://127.0.0.1:11434``) + ``ollama serve``.
- **Ollama Cloud** (``ollama.com``): set ``OLLAMA_HOST=https://ollama.com`` and ``OLLAMA_API_KEY`` (Bearer),
  per https://ollama.com API docs (``/api/generate``).
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
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.api_key = (api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY", "") or "").strip() or None
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    def _is_ollama_com_cloud(self) -> bool:
        """ollama.com hosted API uses ``/api/generate`` + ``Authorization: Bearer``."""
        return bool(self.api_key) and "ollama.com" in self.base_url.lower()

    def _request_json(
        self,
        url: str,
        body: dict,
        *,
        err_label: str,
    ) -> dict:
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace").strip()
                if err_body:
                    detail = f" Body: {err_body[:1200]}{'…' if len(err_body) > 1200 else ''}"
            except Exception:
                pass

            if e.code == 401 or e.code == 403:
                hint = "Unauthorized — check **OLLAMA_API_KEY** (Bearer) for ollama.com, or host access for self-hosted Ollama."
            elif e.code == 404:
                hint = (
                    f"Model missing or wrong name (404). For local Ollama: `ollama list` / `ollama pull {self.model}`. "
                    "For ollama.com, use a model name your account can run."
                )
            elif e.code == 500:
                hint = (
                    "Internal server error — prompt/context too large, or upstream overload. "
                    "Try lowering retrieve_k / max context, or a smaller model."
                )
            else:
                hint = f"Check Ollama / network. Model: {self.model!r}."

            raise RuntimeError(f"Ollama HTTP {e.code} at {err_label} — {e.reason}.{detail} {hint}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}. ({e})") from e

    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        if self._is_ollama_com_cloud():
            # https://ollama.com/api — documented example: POST /api/generate
            url = f"{self.base_url.rstrip('/')}/api/generate"
            body = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_new_tokens, "temperature": 0.2},
            }
            raw = self._request_json(url, body, err_label=url)
            return (raw.get("response") or "").strip()

        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": max_new_tokens, "temperature": 0.2},
        }
        raw = self._request_json(url, body, err_label=f"{self.base_url}/api/chat")
        msg = raw.get("message") or {}
        return (msg.get("content") or "").strip()


def get_generator(
    prefer: str = "ollama",
    *,
    ollama_model: str | None = None,
    ollama_host: str | None = None,
    ollama_api_key: str | None = None,
) -> OllamaChatGenerator:
    """
    Return the Ollama chat generator. Only ``\"ollama\"`` and ``\"auto\"`` are accepted (both use Ollama).
    """
    if prefer not in ("ollama", "auto"):
        raise ValueError(f"Only Ollama is supported; got prefer={prefer!r}")
    return OllamaChatGenerator(
        model=ollama_model,
        base_url=ollama_host,
        api_key=ollama_api_key,
    )


def generator_label(gen: TextGenerator) -> str:
    """Human-readable backend name for logs and UI."""
    if isinstance(gen, OllamaChatGenerator):
        cloud = "+api_key" if gen.api_key else ""
        return f"ollama{cloud}:{gen.model}@{gen.base_url}"
    return type(gen).__name__
