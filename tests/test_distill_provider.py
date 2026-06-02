"""Plan §6: distill provider swap honored; `none` never writes/distills."""

import json
from pathlib import Path

import pytest

from mempalace.memory_miner.distill import distill
from mempalace.memory_miner.providers import ProviderError, get_provider, none_provider


def _transcript(tmp_path: Path) -> Path:
    t = tmp_path / "sess.jsonl"
    rows = [
        {"message": {"role": "user", "content": "Always run pytest before committing."}},
        {"message": {"role": "assistant", "content": "Understood, I'll run the full suite."}},
    ]
    with open(t, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return t


def test_provider_swap_honored(tmp_path):
    captured = {}

    def fake_provider(system, message, *, timeout=180.0, max_tokens=8000):
        captured["system"] = system
        captured["message"] = message
        return json.dumps([
            {"name": "run-pytest", "type": "feedback",
             "description": "Run pytest before commit.",
             "body": "Always run the full test suite before committing.",
             "confidence": 0.9, "likely_duplicate_of": None}
        ])

    out = distill(_transcript(tmp_path), session_id="sess", provider=fake_provider)
    assert len(out) == 1
    assert out[0].type == "feedback"
    assert out[0].source_session == "sess"
    # the transcript reached the (mocked) provider
    assert "pytest" in captured["message"].lower()


def test_none_provider_never_distills():
    with pytest.raises(ProviderError):
        none_provider("sys", "msg")


def test_none_provider_via_distill_yields_nothing(tmp_path):
    # provider 'none' hard-fails inside distill → caught → [] (no crash, no writes)
    out = distill(_transcript(tmp_path), session_id="sess", provider=get_provider("none"))
    assert out == []


def test_get_provider_known_and_unknown():
    assert get_provider("grok").__name__ == "grok_provider"
    with pytest.raises(ValueError):
        get_provider("bogus")
