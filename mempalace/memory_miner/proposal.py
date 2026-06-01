"""
proposal.py — MemoryProposal model, schema validation, stable id, type mapping.

Plan §4.2 (proposal shape + stable id), §4.5 (regex→store type mapping).

A MemoryProposal is the only artifact the distill stage produces. It is
schema-validated before it can reach emit(); malformed proposals are rejected.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# The auto-memory store has exactly 4 types.
STORE_TYPES = ("user", "feedback", "project", "reference")

# Identity types: never auto-updated by dedup; a near-duplicate is flagged for
# human review instead (plan §4.4 / feedback_dont_flip_memory_mid_session).
IDENTITY_TYPES = ("user", "feedback")

# Plan §4.5 — default regex(5)→store(4) mapping. Used as the LLM's prior and as
# the deterministic fallback when provider == "none" (diagnostic).
REGEX_TO_STORE = {
    "preference": "feedback",   # "always/never/I prefer" = how the AI should work
    "decision": "project",      # project state vs durable fact — LLM may override to reference
    "milestone": "project",     # shipped/works = project status
    "problem": "project",       # ongoing problems → project; session-local ones get dropped upstream
    "emotional": "user",        # only durable identity facts; most dropped upstream
}


def map_regex_type(regex_type: Optional[str]) -> str:
    """Map a general_extractor memory_type to a store type (plan §4.5).

    Unknown/None falls back to ``reference`` (the most conservative additive
    bucket). Never raises.
    """
    if not regex_type:
        return "reference"
    return REGEX_TO_STORE.get(regex_type.strip().lower(), "reference")


def slugify(name: str) -> str:
    """kebab/underscore-safe slug from a name. Lowercase, words joined by '_'."""
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower())
    return s.strip("_") or "memory"


def proposal_id(source_session: str, chunk_index: Any, body: str) -> str:
    """Stable content-hash id (plan §4.2): sha256(session + chunk + body)[:16].

    Same source content always yields the same id, so re-runs never emit a
    duplicate proposal line.
    """
    h = hashlib.sha256()
    h.update(str(source_session).encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(str(chunk_index).encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update((body or "").strip().encode("utf-8", "replace"))
    return h.hexdigest()[:16]


@dataclass
class MemoryProposal:
    """A single proposed memory (plan §4.2)."""

    name: str
    type: str
    description: str
    body: str
    source_session: str
    id: str = ""
    source_excerpt: str = ""
    confidence: float = 0.0
    source_chunk: Any = None
    likely_duplicate_of: Optional[str] = None
    # dedup verdict (filled by dedup.py): {action, target, score, enrich}
    # ``enrich`` is True only on a non-identity update → accept.py appends the
    # delta to the existing target instead of creating a duplicate file.
    dedup: dict = field(
        default_factory=lambda: {"action": "create", "target": None, "score": 0.0, "enrich": False}
    )

    def __post_init__(self):
        if not self.id:
            self.id = proposal_id(self.source_session, self.source_chunk, self.body)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryProposal":
        known = {
            "name", "type", "description", "body", "source_session", "id",
            "source_excerpt", "confidence", "source_chunk", "likely_duplicate_of", "dedup",
        }
        return cls(**{k: v for k, v in d.items() if k in known})


# ----------------------------------------------------------------- validation
# Manual JSON-schema (jsonschema not a hard dep — plan allows "jsonschema or manual").
_REQUIRED_STR = ("name", "type", "description", "body")


def validate_proposal(p: Any) -> tuple[bool, str]:
    """Validate a raw distilled dict against the MemoryProposal schema.

    Returns (ok, reason). Reason is empty when ok. Plan §4.2: malformed → reject
    (caller retries once then logs).
    """
    if not isinstance(p, dict):
        return False, "not a dict"
    for k in _REQUIRED_STR:
        v = p.get(k)
        if not isinstance(v, str) or not v.strip():
            return False, f"missing/empty string field: {k}"
    if p["type"] not in STORE_TYPES:
        return False, f"type {p['type']!r} not in {STORE_TYPES}"
    conf = p.get("confidence", 0.0)
    if conf is not None and not isinstance(conf, (int, float)):
        return False, "confidence not numeric"
    if isinstance(conf, (int, float)) and not (0.0 <= float(conf) <= 1.0):
        return False, "confidence out of range [0,1]"
    dup = p.get("likely_duplicate_of")
    if dup is not None and not isinstance(dup, str):
        return False, "likely_duplicate_of not a string/null"
    return True, ""
