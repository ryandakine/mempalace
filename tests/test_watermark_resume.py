"""Plan §6: watermark resume skips first N records; per-file commit persists."""

import json
from pathlib import Path

from mempalace.memory_miner.watermark import Watermark, count_records


def _write_jsonl(path: Path, n: int):
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"message": {"role": "user", "content": f"line {i}"}}) + "\n")


def test_per_file_commit_persisted_to_disk(tmp_path):
    state = tmp_path / "state.json"
    t1 = tmp_path / "s1.jsonl"
    t2 = tmp_path / "s2.jsonl"
    _write_jsonl(t1, 4)
    _write_jsonl(t2, 6)

    wm = Watermark(path=state)
    wm.commit(t1, count_records(t1))
    # state.json exists right after the first file (per-file commit)
    assert state.exists()

    # a fresh Watermark (simulating a resumed process) sees s1 done, s2 pending.
    wm2 = Watermark(path=state)
    assert wm2.needs_processing(t1)[0] is False
    assert wm2.needs_processing(t2)[0] is True


def test_resume_after_abort_continues_at_next_file(tmp_path):
    state = tmp_path / "state.json"
    files = []
    for i in range(5):
        f = tmp_path / f"sess-{i}.jsonl"
        _write_jsonl(f, 3)
        files.append(f)

    wm = Watermark(path=state)
    # simulate abort after committing the first 3 files
    for f in files[:3]:
        wm.commit(f, count_records(f))

    wm2 = Watermark(path=state)
    pending = [f for f in files if wm2.needs_processing(f)[0]]
    assert pending == files[3:]  # resumes at file index 3


def test_entry_records_mtime_size_and_count(tmp_path):
    state = tmp_path / "state.json"
    t = tmp_path / "s.jsonl"
    _write_jsonl(t, 7)
    wm = Watermark(path=state)
    wm.commit(t, 7)
    e = wm.entry("s")
    assert e["records_done"] == 7
    assert e["size"] == t.stat().st_size
    assert "mtime" in e and "last_run" in e
