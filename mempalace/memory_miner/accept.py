"""
accept.py — the ONLY thing that writes a memory file (plan §4.7).

`accept <id>...`  or  `accept --above <conf> --type <t>` promotes queued
proposals into the live memory store:
  - atomic write (temp + os.replace) of the memory .md file
  - IDEMPOTENT MEMORY.md pointer: replace-in-place by slug, never duplicate lines
  - append to the accepted ledger

Default sink = the auto-memory .md store (~/.claude/projects/-home-ryan/memory/).
MemPalace dual-write is OFF by default behind a flag (not implemented in Phase 1
beyond the flag plumbing — Phase 2).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .dedup import slugify

# MEMORY.md section headers per store type.
SECTION = {
    "user": "## User",
    "feedback": "## Feedback",
    "project": "## Project",
    "reference": "## Reference",
}


def default_memory_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".claude" / "projects" / "-home-ryan" / "memory"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def render_memory_md(proposal: Dict) -> str:
    """Render a memory .md with frontmatter matching the store convention."""
    name = proposal.get("name", "").strip()
    desc = proposal.get("description", "").strip()
    mtype = proposal.get("type", "reference").strip()
    body = proposal.get("body", "").strip()
    session = proposal.get("source_session", "").strip()
    fm = ["---", f"name: {name}", f"description: {desc}", f"type: {mtype}"]
    if session:
        fm.append(f"originSessionId: {session}")
    fm.append("---")
    return "\n".join(fm) + "\n\n" + body + "\n"


def memory_filename(proposal: Dict, existing_slugs: set) -> str:
    """`<type>_<slug>.md`, made unique against existing on-disk slugs."""
    base = slugify(proposal.get("name", "memory"))
    mtype = proposal.get("type", "reference")
    stem = f"{mtype}_{base}" if not base.startswith(f"{mtype}_") else base
    candidate = stem
    i = 2
    while f"{candidate}.md" in existing_slugs:
        candidate = f"{stem}_{i}"
        i += 1
    existing_slugs.add(f"{candidate}.md")
    return f"{candidate}.md"


# ----------------------------------------------------------- MEMORY.md pointer
def _short_desc(desc: str, width: int = 100) -> str:
    desc = (desc or "").strip().replace("\n", " ")
    return desc[:width].rstrip()


def update_memory_index(index_path: Path, filename: str, name: str,
                        description: str, store_type: str) -> None:
    """Idempotent MEMORY.md pointer (plan §4.7): replace-in-place by filename slug.

    If a pointer line for ``filename`` already exists anywhere, it is replaced in
    place. Otherwise the new line is appended to the correct `## Type` section
    (creating the section if absent). Never duplicates a line.
    """
    index_path = Path(index_path)
    section = SECTION.get(store_type, "## Reference")
    label = _link_label(name)
    new_line = f"- [{label}]({filename}) — {_short_desc(description)}"

    text = index_path.read_text(encoding="utf-8") if index_path.exists() else "# Memory Index\n"
    lines = text.split("\n")

    # 1. replace-in-place if a pointer to this exact filename already exists.
    link_re = re.compile(r"\]\(" + re.escape(filename) + r"\)")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("- [") and link_re.search(line):
            lines[i] = new_line
            _atomic_write(index_path, "\n".join(lines))
            return

    # 2. else append into the right section (create the section if missing).
    sec_idx = _find_section(lines, section)
    if sec_idx is None:
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append(section)
        lines.append(new_line)
    else:
        insert_at = _section_insert_point(lines, sec_idx)
        lines.insert(insert_at, new_line)

    _atomic_write(index_path, "\n".join(lines))


def _link_label(name: str) -> str:
    """Human label for the MEMORY.md link — lowercase words, no type prefix."""
    label = slugify(name).replace("_", " ").strip()
    return label or name.strip()


def _find_section(lines: List[str], section: str) -> Optional[int]:
    for i, line in enumerate(lines):
        if line.strip() == section:
            return i
    return None


def _section_insert_point(lines: List[str], sec_idx: int) -> int:
    """Index just after the last existing bullet in the section (before next `## `)."""
    i = sec_idx + 1
    last_bullet = sec_idx
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("## "):
            break
        if s.startswith("- ["):
            last_bullet = i
        i += 1
    return last_bullet + 1


# ----------------------------------------------------------------- ledger
def append_ledger(ledger_path: Path, proposal: Dict, filename: str) -> None:
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "id": proposal.get("id"),
        "name": proposal.get("name"),
        "type": proposal.get("type"),
        "file": filename,
        "accepted_at": datetime.now().isoformat(timespec="seconds"),
        "source_session": proposal.get("source_session"),
    }
    with open(ledger_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------- accept driver
def load_queue(jsonl_path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    p = Path(jsonl_path)
    if not p.exists():
        return out
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
                out[d["id"]] = d
    return out


def select_proposals(queue: Dict[str, dict], *, ids: List[str] = None,
                     above: float = None, type_filter: str = None,
                     accepted_ids: set = None) -> List[dict]:
    """Resolve which queued proposals to accept (plan §4.7).

    Explicit ids OR (--above conf [+ --type]). Already-accepted ids are skipped.
    """
    accepted_ids = accepted_ids or set()
    chosen: List[dict] = []
    if ids:
        for i in ids:
            if i in accepted_ids:
                continue
            if i in queue:
                chosen.append(queue[i])
    else:
        for d in queue.values():
            if d["id"] in accepted_ids:
                continue
            if above is not None and float(d.get("confidence") or 0.0) < above:
                continue
            if type_filter is not None and d.get("type") != type_filter:
                continue
            chosen.append(d)
    return chosen


def accept(
    queue_jsonl: Path,
    *,
    ids: List[str] = None,
    above: float = None,
    type_filter: str = None,
    memory_dir: Path = None,
    ledger_path: Path = None,
    index_path: Path = None,
    dual_write_mempalace: bool = False,
) -> List[Dict]:
    """Promote selected proposals into the live store. Returns accepted records.

    The ONLY function in the package that writes a memory file.
    """
    memory_dir = Path(memory_dir) if memory_dir else default_memory_dir()
    ledger_path = Path(ledger_path) if ledger_path else (memory_dir.parent / "memory-miner-accepted.jsonl")
    index_path = Path(index_path) if index_path else (memory_dir / "MEMORY.md")

    from .emit import load_accepted_ids

    queue = load_queue(queue_jsonl)
    accepted_ids = load_accepted_ids(ledger_path)
    chosen = select_proposals(
        queue, ids=ids, above=above, type_filter=type_filter, accepted_ids=accepted_ids
    )

    memory_dir.mkdir(parents=True, exist_ok=True)
    existing_slugs = {f.name for f in memory_dir.glob("*.md")}

    results: List[Dict] = []
    for prop in chosen:
        filename = memory_filename(prop, existing_slugs)
        mem_path = memory_dir / filename
        _atomic_write(mem_path, render_memory_md(prop))
        update_memory_index(
            index_path, filename, prop.get("name", ""),
            prop.get("description", ""), prop.get("type", "reference"),
        )
        append_ledger(ledger_path, prop, filename)
        if dual_write_mempalace:
            _dual_write(prop, filename)
        results.append({"id": prop.get("id"), "file": filename, "type": prop.get("type")})
    return results


def _dual_write(prop: Dict, filename: str) -> None:
    """Optional MemPalace dual-write (OFF by default). Best-effort, never fatal."""
    try:  # pragma: no cover - optional, requires chromadb + palace config
        from ..convo_miner import get_collection
        from ..config import MempalaceConfig

        col = get_collection(MempalaceConfig().palace_path)
        col.add(
            documents=[prop.get("body", "")],
            ids=[f"drawer_memory_{prop.get('id')}"],
            metadatas=[{
                "wing": "memory",
                "room": prop.get("type", "reference"),
                "source_file": filename,
                "added_by": "memory_miner",
                "filed_at": datetime.now().isoformat(),
            }],
        )
    except Exception:
        pass
