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
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


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


def discover_transcripts(roots: List[Path], pattern: str = "*.jsonl") -> List[Path]:
    """Find candidate transcripts under the given roots (sorted, stable)."""
    found: List[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        found.extend(sorted(root.rglob(pattern)))
    # de-dup while preserving order
    seen, out = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out
