"""Plan §6 (CRITICAL): a dry/Phase-0 `run` writes ZERO files to the live store.

`run` may only ever produce the proposal queue. The ONLY thing allowed to write
a memory .md is the gated `accept` command. This test runs the full pipeline
end-to-end with a mocked provider and asserts the memory store is untouched.
"""

import json
from pathlib import Path

import pytest

from mempalace.memory_miner import cli, providers
from mempalace.memory_miner.cli import build_parser


def _make_transcript(d: Path, name: str):
    t = d / name
    rows = [
        {"message": {"role": "user", "content": "Always run pytest before committing."}},
        {"message": {"role": "assistant", "content": "Got it — full suite before every commit."}},
    ]
    with open(t, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return t


def test_run_writes_zero_memory_files(tmp_path, monkeypatch):
    transcripts = tmp_path / "projects"
    transcripts.mkdir()
    _make_transcript(transcripts, "sess-1.jsonl")

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    # seed a pre-existing store + index; the run must not modify either.
    (memory_dir / "MEMORY.md").write_text("# Memory Index\n\n## Reference\n")
    seed = memory_dir / "reference_seed.md"
    seed.write_text("---\nname: Seed\ndescription: d\ntype: reference\n---\nb\n")
    seed_before = seed.read_text()
    index_before = (memory_dir / "MEMORY.md").read_text()

    out_dir = tmp_path / "queue"
    state_file = tmp_path / "state.json"
    lock_file = tmp_path / "miner.lock"

    # mock the provider so NO live call happens; it proposes one memory.
    def fake_grok(system, message, *, timeout=180.0, max_tokens=8000, sleep=None):
        return json.dumps([
            {"name": "run-pytest-before-commit", "type": "feedback",
             "description": "Run pytest before commit.",
             "body": "Always run the full test suite before committing.",
             "confidence": 0.9, "likely_duplicate_of": None}
        ])

    monkeypatch.setitem(providers.PROVIDERS, "grok", fake_grok)
    # redirect the GLOBAL state + lock paths into the temp dir.
    monkeypatch.setattr("mempalace.memory_miner.watermark.state_path", lambda: state_file)
    monkeypatch.setattr("mempalace.memory_miner.watermark.lock_path", lambda: lock_file)

    files_before = {f.name for f in memory_dir.glob("*.md")}

    parser = build_parser()
    args = parser.parse_args([
        "--out-dir", str(out_dir),
        "--memory-dir", str(memory_dir),
        "--ledger", str(tmp_path / "accepted.jsonl"),
        "run", "--provider", "grok", "--root", str(transcripts),
    ])
    rc = args.func(args)
    assert rc == 0

    # ── ZERO writes to the live memory store ──
    files_after = {f.name for f in memory_dir.glob("*.md")}
    assert files_after == files_before, "run created/removed a memory file"
    assert seed.read_text() == seed_before, "run modified an existing memory file"
    assert (memory_dir / "MEMORY.md").read_text() == index_before, "run modified MEMORY.md"
    assert not (tmp_path / "accepted.jsonl").exists(), "run wrote to the accepted ledger"

    # ── but the proposal queue WAS produced ──
    assert (out_dir / "proposals.jsonl").exists()
    assert (out_dir / "proposals.md").exists()
    queued = [json.loads(line) for line in (out_dir / "proposals.jsonl").read_text().splitlines() if line.strip()]
    assert len(queued) == 1
    assert queued[0]["type"] == "feedback"


def test_provider_none_run_writes_nothing(tmp_path, monkeypatch):
    transcripts = tmp_path / "projects"
    transcripts.mkdir()
    _make_transcript(transcripts, "sess-2.jsonl")
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    out_dir = tmp_path / "queue"

    monkeypatch.setattr("mempalace.memory_miner.watermark.state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr("mempalace.memory_miner.watermark.lock_path", lambda: tmp_path / "miner.lock")

    parser = build_parser()
    args = parser.parse_args([
        "--out-dir", str(out_dir), "--memory-dir", str(memory_dir),
        "--ledger", str(tmp_path / "accepted.jsonl"),
        "run", "--provider", "none", "--root", str(transcripts),
    ])
    rc = args.func(args)
    assert rc == 0
    # provider 'none' distills nothing → empty (or absent) queue, no memory files
    assert list(memory_dir.glob("*.md")) == []
    queue = out_dir / "proposals.jsonl"
    if queue.exists():
        assert queue.read_text().strip() == ""
