"""Plan §6: Phase 0 PoC on one transcript asserts ZERO files written.

The PoC (poc_memory_miner.py) is the proven Phase 0 artifact. This runs its
pipeline against a temp transcript with the Grok call mocked and asserts the
filesystem is byte-for-byte unchanged afterward (no writes anywhere under the
temp tree, and main() returns success printing 'ZERO files written').
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
POC = REPO / "poc_memory_miner.py"


def _load_poc():
    spec = importlib.util.spec_from_file_location("poc_memory_miner", POC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _snapshot(root: Path):
    return {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


@pytest.mark.skipif(not POC.exists(), reason="PoC file absent")
def test_poc_run_writes_zero_files(tmp_path, monkeypatch, capsys):
    poc = _load_poc()

    # build a temp transcript
    t = tmp_path / "session.jsonl"
    rows = [
        {"message": {"role": "user", "content": "Always run pytest before committing."}},
        {"message": {"role": "assistant", "content": "Understood — full suite first."}},
    ]
    with open(t, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # point the PoC's memory index at an isolated (empty) dir, never the real store
    fake_mem = tmp_path / "memory"
    fake_mem.mkdir()
    monkeypatch.setattr(poc, "MEM_DIR", fake_mem, raising=False)

    # mock the live Grok call → no network, deterministic JSON
    def fake_grok(system, message, max_tokens=8000, timeout=180.0):
        return json.dumps([
            {"name": "run-pytest", "type": "feedback",
             "description": "Run pytest before commit.",
             "body": "Always run the full test suite before committing.",
             "source_chunk": 0, "confidence": 0.9, "likely_duplicate_of": None}
        ])

    monkeypatch.setattr(poc, "grok", fake_grok)

    before = _snapshot(tmp_path)

    # run the PoC main against the temp transcript
    monkeypatch.setattr("sys.argv", ["poc", "--transcript", str(t), "--mode", "whole"])
    rc = poc.main()

    after = _snapshot(tmp_path)
    # the only file that existed is the transcript; nothing new written, nothing mutated
    assert rc == 0
    assert before == after, "PoC mutated or created files"
    assert list(fake_mem.glob("*")) == [], "PoC wrote into the memory store"

    out = capsys.readouterr().out
    assert "ZERO files written" in out
