"""Plan §6: scan_new — new detected / old skipped / size-shrank → full reparse."""

import json
import os
import time
from pathlib import Path

from mempalace.memory_miner.watermark import (
    Watermark,
    count_records,
    discover_transcripts,
)


def _write_jsonl(path: Path, n: int):
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"message": {"role": "user", "content": f"line {i}"}}) + "\n")


def test_new_file_detected(tmp_path):
    t = tmp_path / "sess-a.jsonl"
    _write_jsonl(t, 5)
    wm = Watermark(path=tmp_path / "state.json")
    should, skip = wm.needs_processing(t)
    assert should is True
    assert skip == 0


def test_unchanged_file_skipped(tmp_path):
    t = tmp_path / "sess-b.jsonl"
    _write_jsonl(t, 5)
    wm = Watermark(path=tmp_path / "state.json")
    wm.commit(t, count_records(t))
    should, _ = wm.needs_processing(t)
    assert should is False


def test_grown_file_resumes_by_record_count(tmp_path):
    t = tmp_path / "sess-c.jsonl"
    _write_jsonl(t, 5)
    wm = Watermark(path=tmp_path / "state.json")
    wm.commit(t, 5)
    # append more records and bump mtime
    with open(t, "a", encoding="utf-8") as fh:
        for i in range(5, 9):
            fh.write(json.dumps({"message": {"role": "user", "content": f"line {i}"}}) + "\n")
    os.utime(t, (time.time() + 5, time.time() + 5))
    should, skip = wm.needs_processing(t)
    assert should is True
    assert skip == 5  # resume by record count, NOT byte offset


def test_size_shrank_triggers_full_reparse(tmp_path):
    t = tmp_path / "sess-d.jsonl"
    _write_jsonl(t, 10)
    wm = Watermark(path=tmp_path / "state.json")
    wm.commit(t, 10)
    _write_jsonl(t, 3)  # rewrite smaller
    os.utime(t, (time.time() + 5, time.time() + 5))
    should, skip = wm.needs_processing(t)
    assert should is True
    assert skip == 0  # full reparse, do NOT trust stale record count


def test_global_state_path_not_project_scoped():
    # Default state path lives under ~/.claude/memory-miner, not any project dir.
    from mempalace.memory_miner.watermark import state_path

    p = state_path()
    assert p.name == "state.json"
    assert p.parent.name == "memory-miner"
    assert ".claude" in str(p)
    assert "projects" not in str(p)


def test_discover_finds_jsonl(tmp_path):
    (tmp_path / "a").mkdir()
    _write_jsonl(tmp_path / "a" / "x.jsonl", 2)
    _write_jsonl(tmp_path / "y.jsonl", 2)
    (tmp_path / "z.txt").write_text("not a transcript")
    found = discover_transcripts([tmp_path])
    names = {p.name for p in found}
    assert names == {"x.jsonl", "y.jsonl"}


def test_count_records_ignores_blank_lines(tmp_path):
    t = tmp_path / "sess-e.jsonl"
    with open(t, "w", encoding="utf-8") as fh:
        fh.write('{"a":1}\n\n{"b":2}\n   \n{"c":3}\n')
    assert count_records(t) == 3
