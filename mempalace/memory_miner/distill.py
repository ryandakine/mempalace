"""
distill.py — whole-session, recall-first distillation (plan §4, §4.1, §4.2).

Phase 0 finding (§1.5): ONE whole-session call with a recall-first prompt and
*synthetic* few-shot examples beats fragmented chunks. ``thinking`` blocks are
signature-only (empty) in CC transcripts, so we use user+assistant text only and
exclude tool noise.

This module:
  1 parses a transcript .jsonl into user/assistant text segments (PoC code)
  2 secret_scrubs the assembled transcript BEFORE the provider sees it
  3 calls the provider (wrapped; hard-fail → [] + log, never crash)
  4 parses the JSON array, schema-validates each item (one retry on malformed)
  5 returns validated MemoryProposal[]
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, List, Optional

from .proposal import MemoryProposal, validate_proposal
from .providers import ProviderError
from .scrub import secret_scrub

log = logging.getLogger("memory_miner.distill")

# Synthetic few-shot — SHAPE ONLY. Real-fact examples get echoed (Phase 0 §1.5 #5).
DISTILL_SYSTEM = """You extract DURABLE facts about the user from a Claude Code session, for a
long-term memory store loaded into every FUTURE session. Be RECALL-oriented: extract every fact that
would still be true or useful next week. Durability/dedup filtering happens downstream — your job is
high recall, not gatekeeping.

The store has exactly 4 types:
- user: who the user is — role, expertise, durable preferences, identity.
- feedback: how the AI should work for the user — corrections, confirmed approaches, workflow rules. Include the WHY.
- project: ongoing work, goals, status, or constraints NOT derivable from code/git.
- reference: pointers to external resources/tools/configs/gotchas (URLs, paths, commands, where-creds-live).

GOOD examples (SHAPE ONLY — these are synthetic; extract the real facts from THIS session, never copy these):
- {"type":"feedback","body":"Run the full test suite before every commit; the user treats a red suite as a hard blocker."}
- {"type":"reference","body":"Production logs live in Grafana Loki — query by the service label."}

RULES:
- One fact per memory. Rewrite into a clean, self-contained statement (a fact about the user / their work /
  their tools — NOT "the assistant did X just now").
- An EXISTING-MEMORIES list is provided. If a fact is already covered, set "likely_duplicate_of" to that
  name; STILL emit it (downstream decides create-vs-update).
- Bodies are pre-redacted as [REDACTED:*] — never invent or restore secrets.
- Output ONLY a JSON array. No prose. No markdown fences.

Each element:
{"name":"kebab-slug","type":"user|feedback|project|reference","description":"one line",
 "body":"the fact","confidence":0.0-1.0,"likely_duplicate_of":"<name or null>"}"""

# A second, stricter nudge appended on the malformed-output retry.
_RETRY_SUFFIX = (
    "\n\nYour previous reply was not a parseable JSON array of the required shape. "
    "Reply with ONLY a valid JSON array. No prose, no markdown fences."
)


def extract_segments(path: Path) -> List[tuple]:
    """Parse a CC .jsonl into [(role, text), ...] keeping user+assistant text only.

    Tool_use / tool_result are excluded. thinking is excluded (signature-only).
    Promoted from poc_memory_miner.extract_segments_raw(include_thinking=False).
    """
    segs: List[tuple] = []
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError as e:
        log.warning("cannot open transcript %s: %s", path, e)
        return segs
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = d.get("message")
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            c = m.get("content")
            if isinstance(c, str):
                if role in ("user", "assistant") and c.strip():
                    segs.append((role, c.strip()))
            elif isinstance(c, list):
                for b in c:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text" and (b.get("text") or "").strip():
                        segs.append((role, b["text"].strip()))
                    # tool_use / tool_result / thinking intentionally excluded
    return segs


def build_transcript(segs: List[tuple]) -> str:
    return "\n\n".join(f"{s.upper()}: {t}" for s, t in segs)


def parse_json_array(raw: str):
    """Extract a JSON array from raw model text (PoC-proven, fence-tolerant)."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw).rstrip("`").strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


def distill(
    transcript_path: Path,
    *,
    session_id: str,
    provider: Callable,
    existing_index: str = "",
    today: str = "",
    char_cap: int = 200_000,
    max_tokens: int = 8000,
    timeout: float = 180.0,
) -> List[MemoryProposal]:
    """Whole-session distill → validated MemoryProposal[]. Never raises.

    ``provider`` is one of providers.PROVIDERS values (already wrapped with
    timeout/backoff). A hard provider failure or unparseable-after-retry output
    yields [] and is logged — the run continues (plan §4.1).
    """
    segs = extract_segments(Path(transcript_path))
    if not segs:
        log.info("no usable segments in %s — skipping", transcript_path)
        return []

    transcript = build_transcript(segs)
    # secret_scrub BEFORE the provider (and before any disk write of excerpts).
    scrubbed, _redacted = secret_scrub(transcript[:char_cap])

    date_line = f"TODAY IS {today}.\n\n" if today else ""
    base_message = (
        f"{date_line}EXISTING MEMORIES (name: description):\n{existing_index}\n\n"
        f"{'=' * 40}\nSESSION TRANSCRIPT:\n{scrubbed}\n\n"
        f"Return the JSON array now."
    )

    raw = _call_provider(provider, base_message, max_tokens, timeout)
    if raw is None:
        return []

    proposals = parse_json_array(raw)
    if proposals is None or not isinstance(proposals, list):
        # one retry with a stricter nudge (plan §4.2)
        log.info("distill output unparseable for %s — retrying once", session_id)
        raw = _call_provider(provider, base_message + _RETRY_SUFFIX, max_tokens, timeout)
        if raw is None:
            return []
        proposals = parse_json_array(raw)
        if proposals is None or not isinstance(proposals, list):
            log.warning("distill output still unparseable for %s — rejecting", session_id)
            return []

    out: List[MemoryProposal] = []
    for p in proposals:
        ok, reason = validate_proposal(p)
        if not ok:
            log.info("rejected proposal in %s: %s", session_id, reason)
            continue
        out.append(
            MemoryProposal(
                name=p["name"].strip(),
                type=p["type"],
                description=p["description"].strip(),
                body=p["body"].strip(),
                source_session=session_id,
                source_excerpt=_excerpt(p["body"], scrubbed),
                confidence=float(p.get("confidence") or 0.0),
                source_chunk=p.get("source_chunk"),
                likely_duplicate_of=p.get("likely_duplicate_of") or None,
            )
        )
    return out


def _call_provider(provider: Callable, message: str, max_tokens: int,
                   timeout: float) -> Optional[str]:
    try:
        return provider(DISTILL_SYSTEM, message, timeout=timeout, max_tokens=max_tokens)
    except ProviderError as e:
        log.warning("provider hard-failed, skipping: %s", e)
        return None
    except Exception as e:  # defensive: never crash the run on a provider quirk
        log.warning("provider unexpected error, skipping: %r", e)
        return None


def _excerpt(body: str, transcript: str, width: int = 240) -> str:
    """Best-effort short source excerpt for human review. Already-scrubbed input."""
    return (body or "").strip()[:width]
