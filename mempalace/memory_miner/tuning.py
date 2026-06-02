"""
tuning.py — decision-logging scaffold for self-tuning (Task #6, Phase 1).

This module CAPTURES the human/operator decisions made on proposals (accept,
reject, merge, review) into an append-only JSONL ledger at
``~/.claude/memory-miner/decisions.jsonl``. It is the data-capture foundation
for self-tuning — nothing more.

Phase 2 will CONSUME decisions.jsonl to tune the distill prompt / dedup
thresholds / type mapping. Not implemented yet — this only CAPTURES the signal.

Design notes:
  - Append-only: every decision is a new line; we never rewrite the ledger.
  - Library code stays deterministic-friendly: callers may pass an explicit
    ``ts`` (or omit it entirely). We do NOT stamp wall-clock by default, so
    tests and re-runs are reproducible.
  - ``decided_ids`` lets future runs / emit skip proposals that already have a
    recorded decision (the integration wiring for that lives outside this
    module — see the module owner's notes).

TODO(phase2): build the tuning algorithm on top of this ledger:
  - mine accepted/rejected ratios per regex_type → adjust REGEX_TO_STORE prior
  - mine merged vs created → adjust dedup thresholds
  - feed accept/reject excerpts back into the distill prompt as few-shot signal
  Do NOT build it until there is enough captured signal to learn from.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger("memory_miner.tuning")

# Recognized decision actions. Kept permissive (we don't reject unknowns) so the
# scaffold never blocks a caller, but these are the canonical set.
ACTIONS = ("accepted", "rejected", "merged", "reviewed")


def default_decisions_path() -> Path:
    """Global decisions ledger, alongside the rest of memory-miner state."""
    return (
        Path(os.path.expanduser("~"))
        / ".claude"
        / "memory-miner"
        / "decisions.jsonl"
    )


def _resolve(ledger_path: Optional[Path]) -> Path:
    return Path(ledger_path) if ledger_path else default_decisions_path()


def record_decision(
    decision_id: str,
    action: str,
    *,
    type: Optional[str] = None,
    score: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
    ts: Optional[str] = None,
    ledger_path: Optional[Path] = None,
) -> bool:
    """Append one decision to the decisions ledger.

    ``action`` should be one of ``ACTIONS`` ({accepted, rejected, merged,
    reviewed}); unknown actions are still recorded (with a debug log) so the
    scaffold never drops signal.

    No wall-clock is stamped by default — pass ``ts`` explicitly if you want a
    timestamp; otherwise the ``ts`` key is omitted entirely (keeps library code
    deterministic for tests/re-runs).

    Returns True if a line was written, False on any failure (best-effort: the
    caller's primary work must never break because logging failed).
    """
    if not decision_id:
        return False
    if action not in ACTIONS:
        log.debug("record_decision: non-canonical action %r", action)

    rec: Dict[str, Any] = {"id": decision_id, "action": action}
    if type is not None:
        rec["type"] = type
    if score is not None:
        rec["score"] = score
    if ts is not None:
        rec["ts"] = ts
    if meta:
        rec["meta"] = meta

    path = _resolve(ledger_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception:  # best-effort — never fatal
        log.debug("record_decision: failed to write decision %r", decision_id,
                  exc_info=True)
        return False


def reject(decision_id: str, ledger_path: Optional[Path] = None,
           **meta: Any) -> bool:
    """Convenience: record a rejection. Extra kwargs land in ``meta``.

    ``type``, ``score``, and ``ts`` are pulled out of the kwargs and passed as
    first-class fields; everything else becomes ``meta``.
    """
    type_ = meta.pop("type", None)
    score = meta.pop("score", None)
    ts = meta.pop("ts", None)
    return record_decision(
        decision_id, "rejected",
        type=type_, score=score, ts=ts,
        meta=meta or None, ledger_path=ledger_path,
    )


def load_decisions(ledger_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read every decision record. Tolerant of blank/malformed lines."""
    path = _resolve(ledger_path)
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and d.get("id"):
                out.append(d)
    return out


def decided_ids(ledger_path: Optional[Path] = None) -> Set[str]:
    """Set of proposal ids that already have a recorded decision.

    Future runs / emit can use this to skip proposals a human has already
    acted on (accept/reject/merge/review), so they don't resurface.
    """
    return {d["id"] for d in load_decisions(ledger_path) if d.get("id")}
