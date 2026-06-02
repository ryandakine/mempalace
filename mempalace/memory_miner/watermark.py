"""
watermark.py — global, record-count incremental state + whole-run flock (plan §4.3).

State file is GLOBAL: ~/.claude/memory-miner/state.json (NOT under any project
dir). Key = transcript filename stem. Value = {mtime, size, records_done, last_run}.

Resume is by RECORD COUNT — skip the first ``records_done`` JSON lines, never byte
offsets. If size shrank (truncate/rewrite) → full reparse. Watermark is committed
per-file so an abort mid-backlog resumes at the next file.

A non-blocking whole-run flock is acquired at run START; a second concurrent run
exits immediately.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


def parse_since(value: str) -> float:
    """Parse a ``YYYY-MM-DD`` --since value → POSIX timestamp at local midnight.

    Raises ``ValueError`` with an actionable message on malformed input so the
    CLI can surface a clean error (not a traceback).
    """
    try:
        dt = datetime.strptime(value.strip(), "%Y-%m-%d")
    except (ValueError, AttributeError) as e:
        raise ValueError(
            f"invalid --since {value!r}: expected YYYY-MM-DD (e.g. 2026-05-01)"
        ) from e
    return dt.timestamp()


def state_dir() -> Path:
    """Global state dir, honoring HOME (so tests with patched HOME are isolated)."""
    return Path(os.path.expanduser("~")) / ".claude" / "memory-miner"


def state_path() -> Path:
    return state_dir() / "state.json"


def lock_path() -> Path:
    return state_dir() / "miner.lock"


class AlreadyRunning(Exception):
    """Raised when a second concurrent run cannot acquire the whole-run lock."""


@contextmanager
def run_lock(path: Optional[Path] = None) -> Iterator[None]:
    """Non-blocking whole-run flock (plan §4.3). Second run → AlreadyRunning.

    Guards discovery, not just the state write, by being acquired at run start.
    """
    p = Path(path) if path else lock_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "w")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                raise AlreadyRunning(f"another miner run holds {p}") from e
            raise
        try:
            fh.write(str(os.getpid()))
            fh.flush()
        except OSError:
            pass
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


class Watermark:
    """Load/inspect/commit the global record-count watermark."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else state_path()
        self.data: Dict[str, dict] = self._load()

    def _load(self) -> Dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8")) or {}
        except (OSError, json.JSONDecodeError):
            return {}

    def entry(self, stem: str) -> Optional[dict]:
        return self.data.get(stem)

    def needs_processing(self, transcript: Path) -> Tuple[bool, int]:
        """(should_process, skip_records).

        - new file → (True, 0)
        - mtime/size changed and size grew → (True, records_done)  [resume]
        - size shrank → (True, 0)  [full reparse — append-only assumption broken]
        - unchanged → (False, 0)
        """
        transcript = Path(transcript)
        try:
            st = transcript.stat()
        except OSError:
            return False, 0
        prev = self.data.get(transcript.stem)
        if prev is None:
            return True, 0
        if prev.get("mtime") == st.st_mtime and prev.get("size") == st.st_size:
            return False, 0
        if st.st_size < int(prev.get("size", 0)):
            # truncate / rewrite → cannot trust record offset
            return True, 0
        return True, int(prev.get("records_done", 0))

    def commit(self, transcript: Path, records_done: int) -> None:
        """Commit watermark for ONE file and flush to disk (atomic) — plan §4.3."""
        transcript = Path(transcript)
        try:
            st = transcript.stat()
        except OSError:
            return
        self.data[transcript.stem] = {
            "mtime": st.st_mtime,
            "size": st.st_size,
            "records_done": int(records_done),
            "last_run": time.time(),
        }
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


def count_records(transcript: Path) -> int:
    """Count non-blank JSON lines (record count) in a transcript."""
    n = 0
    try:
        with open(transcript, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    n += 1
    except OSError:
        return 0
    return n


_SUBTRANSCRIPT_DIRS = {"subagents", "workflows"}


def _is_subtranscript(p: Path) -> bool:
    """True for subagent transcripts / workflow journals — implementation chatter,
    not durable user sessions. Real sessions are ``<uuid>.jsonl`` directly under a
    project dir; subagent runs are ``…/subagents/agent-*.jsonl`` and workflow runs
    are ``…/subagents/workflows/wf_*/journal.jsonl``."""
    if p.name.startswith("agent-") or p.name == "journal.jsonl":
        return True
    return any(part in _SUBTRANSCRIPT_DIRS for part in p.parts)


def discover_transcripts(
    roots: List[Path],
    pattern: str = "*.jsonl",
    *,
    since_ts: Optional[float] = None,
    newest_first: bool = False,
) -> List[Path]:
    """Find candidate transcripts under the given roots (sorted, stable).

    Backlog cost controls (the live backlog is ~1.6k transcripts):
      - ``since_ts``: keep only files with mtime >= this POSIX timestamp.
        Files whose stat() fails are dropped (cannot prove they qualify).
      - ``newest_first``: order by mtime descending instead of by path, so a
        capped (``--limit``) run hits the most RECENT sessions, not arbitrary
        os.walk-order ones. Ties broken by path for stability.

    Compaction note: compaction-continuation transcripts (a session resumed
    after auto-compact lands in a fresh file that re-states earlier context)
    are NOT de-duped here. They are distinct files with distinct stems, so the
    record-count watermark treats each independently. A future enhancement
    could group continuations by a shared session/lineage id embedded in the
    transcript header and skip re-mining the carried-over prefix; that belongs
    in the distill/dedup layer, not in plain file discovery, so it is left out
    deliberately to avoid over-engineering discovery.
    """
    found: List[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        found.extend(sorted(root.rglob(pattern)))
    # de-dup while preserving order
    seen, out = set(), []
    for p in found:
        if p in seen or _is_subtranscript(p):
            continue
        seen.add(p)
        out.append(p)

    if since_ts is None and not newest_first:
        return out

    # Annotate with mtime once; files we cannot stat are excluded when filtering
    # by --since (we can't prove they qualify) and sorted last when newest-first.
    annotated: List[Tuple[float, Path]] = []
    for p in out:
        try:
            mtime = p.stat().st_mtime
        except OSError:
            if since_ts is not None:
                continue  # cannot prove mtime >= cutoff → drop
            mtime = float("-inf")
        if since_ts is not None and mtime < since_ts:
            continue
        annotated.append((mtime, p))

    if newest_first:
        # mtime desc, path asc for deterministic tie-breaking
        annotated.sort(key=lambda mp: (-mp[0], str(mp[1])))
        return [p for _, p in annotated]

    return [p for _, p in annotated]
