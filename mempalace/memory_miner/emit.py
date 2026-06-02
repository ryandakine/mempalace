"""
emit.py — write the proposal queue (plan §4.7).

proposals.jsonl : one MemoryProposal dict per line, keyed by stable ``id``.
proposals.md    : human-readable review queue.

Re-runs must NOT append duplicate proposals: an id already present in
proposals.jsonl OR already in the accepted ledger is skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Set

from .proposal import MemoryProposal


def load_existing_ids(jsonl_path: Path) -> Set[str]:
    ids: Set[str] = set()
    p = Path(jsonl_path)
    if not p.exists():
        return ids
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and d.get("id"):
                ids.add(d["id"])
    return ids


def load_accepted_ids(ledger_path: Path) -> Set[str]:
    """Accepted-ledger ids. Ledger is JSONL; tolerate plain-id lines too."""
    ids: Set[str] = set()
    p = Path(ledger_path)
    if not p.exists():
        return ids
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if isinstance(d, dict) and d.get("id"):
                    ids.add(d["id"])
                    continue
            except json.JSONDecodeError:
                pass
            ids.add(line)
    return ids


def emit(
    proposals: Iterable[MemoryProposal],
    out_dir: Path,
    *,
    accepted_ledger: Path = None,
    decisions_ledger: Path = None,
) -> List[MemoryProposal]:
    """Append NEW proposals to proposals.jsonl and rewrite proposals.md.

    Returns the proposals that were newly written (skipping ids already present
    in the queue, the accepted ledger, or the decisions ledger — so a proposal
    you rejected/reviewed/accepted never resurfaces on a later run). proposals.md
    is regenerated from the full jsonl so it always reflects the current queue.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "proposals.jsonl"
    md_path = out_dir / "proposals.md"

    seen = load_existing_ids(jsonl_path)
    if accepted_ledger is not None:
        seen |= load_accepted_ids(accepted_ledger)
    if decisions_ledger is not None:
        from .tuning import decided_ids
        seen |= decided_ids(decisions_ledger)

    newly: List[MemoryProposal] = []
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        for p in proposals:
            if p.id in seen:
                continue
            seen.add(p.id)
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
            newly.append(p)

    _write_md(jsonl_path, md_path)
    return newly


def _truncate_excerpt(text: str, width: int = 200) -> str:
    """Collapse whitespace and clip the provenance excerpt for the review queue."""
    text = " ".join((text or "").split())
    if len(text) > width:
        text = text[:width].rstrip() + "…"
    return text


def _write_md(jsonl_path: Path, md_path: Path) -> None:
    rows = []
    if Path(jsonl_path).exists():
        with open(jsonl_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    lines = ["# Memory Miner — Proposal Queue", ""]
    lines.append(f"_{len(rows)} proposal(s). Review, then `accept <id>` to promote._")
    lines.append("")
    by_action = {}
    for r in rows:
        action = (r.get("dedup") or {}).get("action", "create")
        by_action.setdefault(action, []).append(r)

    for action in ("create", "update", "review", "skip"):
        group = by_action.get(action, [])
        if not group:
            continue
        lines.append(f"## {action} ({len(group)})")
        lines.append("")
        for r in group:
            ded = r.get("dedup") or {}
            target = ded.get("target")
            tgt = f" → `{target}`" if target else ""
            # Staleness marker: ⏳ flags facts the heuristic judged time-sensitive
            # (in-flight project state) — re-verify before accepting.
            stale = " ⏳" if r.get("time_sensitive") else ""
            lines.append(
                f"### `{r.get('id', '?')}`  [{r.get('type', '?')}] {r.get('name', '?')}{stale}"
                f"  (conf={r.get('confidence', '?')}, dedup={ded.get('score', '?')}{tgt})"
            )
            lines.append("")
            if r.get("description"):
                lines.append(f"_{r['description']}_")
                lines.append("")
            lines.append("> " + (r.get("body", "").strip().replace("\n", "\n> ")))
            lines.append("")
            excerpt = (r.get("source_excerpt") or "").strip()
            if excerpt:
                excerpt = _truncate_excerpt(excerpt)
                lines.append("> excerpt: " + excerpt.replace("\n", " "))
                lines.append("")
            lines.append(f"<sub>source: {r.get('source_session', '?')}</sub>")
            lines.append("")

    tmp = Path(str(md_path) + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, md_path)
