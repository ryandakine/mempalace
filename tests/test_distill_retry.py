"""Plan §6: distill output unparseable → one retry → then reject (no crash)."""

import json
from pathlib import Path

from mempalace.memory_miner.distill import distill


def _transcript(tmp_path: Path) -> Path:
    t = tmp_path / "sess.jsonl"
    rows = [
        {"message": {"role": "user", "content": "Ryan deploys via Docker Compose."}},
        {"message": {"role": "assistant", "content": "Noted."}},
    ]
    with open(t, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return t


def test_retry_recovers_on_second_attempt(tmp_path):
    calls = {"n": 0}

    def flaky(system, message, *, timeout=180.0, max_tokens=8000):
        calls["n"] += 1
        if calls["n"] == 1:
            return "sorry, here is some prose with no JSON array"
        return json.dumps([
            {"name": "docker-deploy", "type": "reference",
             "description": "Deploys via Docker Compose.",
             "body": "Ryan deploys all projects via Docker Compose.",
             "confidence": 0.8}
        ])

    out = distill(_transcript(tmp_path), session_id="sess", provider=flaky)
    assert calls["n"] == 2  # retried exactly once
    assert len(out) == 1


def test_reject_after_retry_still_unparseable(tmp_path):
    calls = {"n": 0}

    def always_bad(system, message, *, timeout=180.0, max_tokens=8000):
        calls["n"] += 1
        return "no json here at all"

    out = distill(_transcript(tmp_path), session_id="sess", provider=always_bad)
    assert calls["n"] == 2  # one initial + one retry
    assert out == []  # rejected, not crashed


def test_individual_malformed_items_dropped_valid_kept(tmp_path):
    def mixed(system, message, *, timeout=180.0, max_tokens=8000):
        return json.dumps([
            {"name": "good", "type": "reference", "description": "d", "body": "b", "confidence": 0.7},
            {"name": "bad", "type": "not-a-type", "description": "d", "body": "b"},
            {"name": "", "type": "reference", "description": "d", "body": "b"},  # empty name
        ])

    out = distill(_transcript(tmp_path), session_id="sess", provider=mixed)
    assert len(out) == 1
    assert out[0].name == "good"
