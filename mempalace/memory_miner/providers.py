"""
providers.py — pluggable distill providers (plan §4.1).

A provider is a callable ``(system, message, *, timeout, max_tokens) -> str``
that returns raw model text (expected to contain a JSON array). Every provider
call is wrapped with:
  - per-call timeout
  - 429 / 5xx retry-with-backoff
  - hard-failure → raise ProviderError (caller skips + logs, never crashes the run)

Providers:
  grok   (default) — local xAI proxy at http://127.0.0.1:8765/ask_grok (PoC-proven)
  local            — local OpenAI-compatible endpoint (rig/Fold6), $0
  claude           — ~/bin/ask-claude (Anthropic API, OFF the CC subscription)
  none             — diagnostic only; raises if asked to generate (never writes)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

GROK_PROXY = os.environ.get("GROK_PROXY_URL", "http://127.0.0.1:8765/ask_grok")
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:11434/v1/chat/completions")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "gemma2")
ASK_CLAUDE_BIN = os.environ.get("ASK_CLAUDE_BIN", os.path.expanduser("~/bin/ask-claude"))

RETRYABLE_HTTP = {429, 500, 502, 503, 504}


class ProviderError(Exception):
    """Raised when a provider hard-fails after retries. Caller skips + logs."""


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _with_retry(fn, *, retries: int = 3, base_delay: float = 1.5, sleep=time.sleep):
    """Call fn(); retry on 429/5xx and transient URLErrors with exponential backoff.

    Non-retryable HTTP (e.g. 400/401) raises ProviderError immediately.
    ``sleep`` is injectable so tests don't actually wait.
    """
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in RETRYABLE_HTTP:
                raise ProviderError(f"HTTP {e.code}: {_read_err(e)}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        except json.JSONDecodeError as e:
            # malformed transport-level JSON — treat as transient once
            last = e
        if attempt < retries - 1:
            sleep(base_delay * (2 ** attempt))
    raise ProviderError(f"provider failed after {retries} attempts: {last!r}")


def _read_err(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode(errors="replace")[:300]
    except Exception:  # pragma: no cover
        return ""


# ------------------------------------------------------------------ providers
def grok_provider(system: str, message: str, *, timeout: float = 180.0,
                  max_tokens: int = 8000, sleep=time.sleep) -> str:
    """Distill via the local Grok proxy (default). Mirrors poc_memory_miner.grok()."""
    def _call():
        body = _http_post_json(
            GROK_PROXY,
            {"message": message, "system": system, "max_tokens": max_tokens},
            timeout,
        )
        return body.get("content") or ""

    return _with_retry(_call, sleep=sleep)


def local_provider(system: str, message: str, *, timeout: float = 300.0,
                   max_tokens: int = 8000, sleep=time.sleep) -> str:
    """Distill via a local OpenAI-compatible chat endpoint (Ollama/llama.cpp). $0."""
    def _call():
        body = _http_post_json(
            LOCAL_LLM_URL,
            {
                "model": LOCAL_LLM_MODEL,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": message},
                ],
            },
            timeout,
        )
        choices = body.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""

    return _with_retry(_call, sleep=sleep)


def claude_provider(system: str, message: str, *, timeout: float = 180.0,
                    max_tokens: int = 8000, sleep=time.sleep) -> str:
    """Distill via ~/bin/ask-claude (Anthropic API, pay-as-you-go, off CC sub)."""
    def _call():
        if not os.path.exists(ASK_CLAUDE_BIN):
            raise ProviderError(f"ask-claude not found at {ASK_CLAUDE_BIN}")
        try:
            proc = subprocess.run(
                [ASK_CLAUDE_BIN, "--system", system, "--max-tokens", str(max_tokens), message],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(str(e)) from e
        if proc.returncode != 0:
            raise ProviderError(f"ask-claude exit {proc.returncode}: {proc.stderr[:300]}")
        return proc.stdout or ""

    return _with_retry(_call, sleep=sleep)


def none_provider(system: str, message: str, **_kw) -> str:
    """Diagnostic provider — never generates, never writes (plan §4.1)."""
    raise ProviderError("provider 'none' is diagnostic-only and never distills")


PROVIDERS = {
    "grok": grok_provider,
    "local": local_provider,
    "claude": claude_provider,
    "none": none_provider,
}


def get_provider(name: str):
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}")
    return PROVIDERS[name]
