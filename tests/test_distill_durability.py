"""Plan §6: distill returns None/empty on session-local chatter."""

import json
from pathlib import Path

from mempalace.memory_miner.distill import distill, extract_segments


def _transcript(tmp_path: Path, rows) -> Path:
    t = tmp_path / "sess.jsonl"
    with open(t, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return t


def test_empty_array_from_provider_yields_no_proposals(tmp_path):
    t = _transcript(tmp_path, [
        {"message": {"role": "user", "content": "thanks!"}},
        {"message": {"role": "assistant", "content": "you're welcome"}},
    ])

    def chatter_provider(system, message, *, timeout=180.0, max_tokens=8000):
        return "[]"  # provider judged nothing durable

    out = distill(t, session_id="sess", provider=chatter_provider)
    assert out == []


def test_no_usable_segments_skips_without_calling_provider(tmp_path):
    # only tool noise / no user-or-assistant text → distill returns [] early.
    t = _transcript(tmp_path, [
        {"message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}}
        ]}},
        {"message": {"role": "user", "content": [
            {"type": "tool_result", "content": "file1 file2"}
        ]}},
    ])
    called = {"n": 0}

    def provider(system, message, *, timeout=180.0, max_tokens=8000):
        called["n"] += 1
        return "[]"

    out = distill(t, session_id="sess", provider=provider)
    assert out == []
    assert called["n"] == 0  # provider never invoked when there's nothing to distill


def test_extract_segments_excludes_thinking_and_tools(tmp_path):
    t = _transcript(tmp_path, [
        {"message": {"role": "user", "content": "real user text"}},
        {"message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "secret reasoning"},
            {"type": "text", "text": "real assistant text"},
            {"type": "tool_use", "name": "Bash", "input": {}},
        ]}},
    ])
    segs = extract_segments(t)
    texts = [s[1] for s in segs]
    assert "real user text" in texts
    assert "real assistant text" in texts
    assert "secret reasoning" not in texts
