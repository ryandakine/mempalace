"""
scrub.py — deterministic secret redaction (plan §4.6).

Applied to every chunk BEFORE it reaches any provider or disk. Never delegated
to LLM behavior. Patterns promoted verbatim from poc_memory_miner.py (proven in
Phase 0).
"""

from __future__ import annotations

import re

# Deterministic, pre-provider. Plan §4.6 — never delegated to the LLM.
# (Promoted unchanged from poc_memory_miner.py SECRET_PATTERNS.)
SECRET_PATTERNS = [
    ("anthropic", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("aws", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("bearer", re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}")),
    (
        "assign",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key)\b"
            r"\s*[=:]\s*['\"]?([A-Za-z0-9/_+\-]{12,})['\"]?"
        ),
    ),
    # long opaque blob — never needed verbatim in a memory body
    ("blob", re.compile(r"\b[A-Za-z0-9+/=_\-]{40,}\b")),
]


def secret_scrub(text: str) -> tuple[str, int]:
    """Redact secret-shaped substrings. Returns (scrubbed_text, n_redactions).

    Deterministic and idempotent on already-redacted text (the placeholder
    contains no secret-shaped tokens).
    """
    if not text:
        return text, 0
    n = 0
    for label, pat in SECRET_PATTERNS:

        def _sub(m, label=label):
            nonlocal n
            n += 1
            return f"[REDACTED:{label}]"

        text = pat.sub(_sub, text)
    return text, n
