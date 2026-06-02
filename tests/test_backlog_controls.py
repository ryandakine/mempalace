"""Task #4: backlog cost controls — --since, --newest-first, real local provider.

All tests mock everything (no live HTTP, no real LLM), use temp dirs, and
control mtimes with os.utime so ordering/filtering is deterministic.
"""

import json
import os
import time
import urllib.error
from datetime import datetime
from pathlib import Path

import pytest

from mempalace.memory_miner import providers
from mempalace.memory_miner.providers import ProviderError
from mempalace.memory_miner.watermark import discover_transcripts, parse_since


# --------------------------------------------------------------------- helpers
def _touch_jsonl(path: Path, mtime: float, n: int = 2) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"message": {"role": "user", "content": f"l{i}"}}) + "\n")
    os.utime(path, (mtime, mtime))
    return path


def _ts(y, m, d) -> float:
    return datetime(y, m, d).timestamp()


# --------------------------------------------------------------- parse_since
def test_parse_since_valid():
    assert parse_since("2026-05-01") == datetime(2026, 5, 1).timestamp()


def test_parse_since_strips_whitespace():
    assert parse_since("  2026-05-01  ") == datetime(2026, 5, 1).timestamp()


@pytest.mark.parametrize("bad", ["2026-13-01", "not-a-date", "05/01/2026", "", "2026-5"])
def test_parse_since_bad_raises_value_error(bad):
    with pytest.raises(ValueError) as ei:
        parse_since(bad)
    assert "YYYY-MM-DD" in str(ei.value)


# ------------------------------------------------------------------- --since
def test_since_filters_by_mtime(tmp_path):
    old = _touch_jsonl(tmp_path / "old.jsonl", _ts(2026, 1, 1))
    mid = _touch_jsonl(tmp_path / "mid.jsonl", _ts(2026, 5, 15))
    new = _touch_jsonl(tmp_path / "new.jsonl", _ts(2026, 5, 30))

    found = discover_transcripts([tmp_path], since_ts=_ts(2026, 5, 1))
    names = {p.name for p in found}
    assert names == {"mid.jsonl", "new.jsonl"}
    assert old not in found
    assert mid in found and new in found


def test_since_boundary_is_inclusive(tmp_path):
    cutoff = _ts(2026, 5, 1)
    on = _touch_jsonl(tmp_path / "on.jsonl", cutoff)
    just_before = _touch_jsonl(tmp_path / "before.jsonl", cutoff - 1)
    found = discover_transcripts([tmp_path], since_ts=cutoff)
    assert on in found
    assert just_before not in found


def test_since_none_returns_everything(tmp_path):
    _touch_jsonl(tmp_path / "a.jsonl", _ts(2020, 1, 1))
    _touch_jsonl(tmp_path / "b.jsonl", _ts(2026, 5, 1))
    found = discover_transcripts([tmp_path])  # since_ts default None
    assert {p.name for p in found} == {"a.jsonl", "b.jsonl"}


# ------------------------------------------------------------ --newest-first
def test_newest_first_orders_by_mtime_desc(tmp_path):
    _touch_jsonl(tmp_path / "a.jsonl", _ts(2026, 1, 1))
    _touch_jsonl(tmp_path / "b.jsonl", _ts(2026, 5, 30))
    _touch_jsonl(tmp_path / "c.jsonl", _ts(2026, 3, 15))

    found = discover_transcripts([tmp_path], newest_first=True)
    assert [p.name for p in found] == ["b.jsonl", "c.jsonl", "a.jsonl"]


def test_default_order_is_path_sorted_not_mtime(tmp_path):
    # Without newest_first, order is os.walk/path-sorted regardless of mtime.
    _touch_jsonl(tmp_path / "a.jsonl", _ts(2026, 5, 30))  # newest but name 'a'
    _touch_jsonl(tmp_path / "z.jsonl", _ts(2026, 1, 1))   # oldest but name 'z'
    found = discover_transcripts([tmp_path])
    assert [p.name for p in found] == ["a.jsonl", "z.jsonl"]


def test_newest_first_with_limit_picks_recent(tmp_path):
    """Simulate the cmd_run cap: newest-first + take first N => most recent N."""
    _touch_jsonl(tmp_path / "old1.jsonl", _ts(2026, 1, 1))
    _touch_jsonl(tmp_path / "old2.jsonl", _ts(2026, 2, 1))
    _touch_jsonl(tmp_path / "recent1.jsonl", _ts(2026, 5, 28))
    _touch_jsonl(tmp_path / "recent2.jsonl", _ts(2026, 5, 30))

    ordered = discover_transcripts([tmp_path], newest_first=True)
    capped = ordered[:2]  # what cmd_run's --limit would mine
    assert {p.name for p in capped} == {"recent2.jsonl", "recent1.jsonl"}


def test_newest_first_ties_broken_by_path(tmp_path):
    same = _ts(2026, 5, 1)
    _touch_jsonl(tmp_path / "b.jsonl", same)
    _touch_jsonl(tmp_path / "a.jsonl", same)
    found = discover_transcripts([tmp_path], newest_first=True)
    assert [p.name for p in found] == ["a.jsonl", "b.jsonl"]


def test_since_and_newest_first_combined(tmp_path):
    _touch_jsonl(tmp_path / "old.jsonl", _ts(2026, 1, 1))
    _touch_jsonl(tmp_path / "m1.jsonl", _ts(2026, 5, 10))
    _touch_jsonl(tmp_path / "m2.jsonl", _ts(2026, 5, 25))
    found = discover_transcripts(
        [tmp_path], since_ts=_ts(2026, 5, 1), newest_first=True
    )
    assert [p.name for p in found] == ["m2.jsonl", "m1.jsonl"]


# ----------------------------------------------------------- local provider
def test_local_provider_unset_url_raises_provider_error(monkeypatch):
    monkeypatch.delenv("MINER_LOCAL_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    with pytest.raises(ProviderError) as ei:
        providers.local_provider("sys", "msg", sleep=lambda s: None)
    # actionable message names the env var and never makes a network call
    assert "MINER_LOCAL_URL" in str(ei.value)


def test_local_provider_unset_url_does_not_hit_network(monkeypatch):
    monkeypatch.delenv("MINER_LOCAL_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("must not POST when URL unset")

    monkeypatch.setattr(providers, "_http_post_json", boom)
    with pytest.raises(ProviderError):
        providers.local_provider("sys", "msg", sleep=lambda s: None)
    assert called["n"] == 0


def test_local_provider_hits_mocked_endpoint(monkeypatch):
    monkeypatch.setenv("MINER_LOCAL_URL", "http://local.test/v1/chat/completions")
    monkeypatch.setenv("MINER_LOCAL_MODEL", "test-model")
    seen = {}

    def fake_post(url, payload, timeout):
        seen["url"] = url
        seen["payload"] = payload
        seen["timeout"] = timeout
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr(providers, "_http_post_json", fake_post)
    out = providers.local_provider("the-system", "the-msg", timeout=42.0, sleep=lambda s: None)
    assert out == "[]"
    assert seen["url"] == "http://local.test/v1/chat/completions"
    assert seen["timeout"] == 42.0
    assert seen["payload"]["model"] == "test-model"
    # OpenAI-compatible message shape
    roles = [m["role"] for m in seen["payload"]["messages"]]
    assert roles == ["system", "user"]
    assert seen["payload"]["messages"][0]["content"] == "the-system"
    assert seen["payload"]["messages"][1]["content"] == "the-msg"


def test_local_provider_falls_back_to_legacy_env(monkeypatch):
    monkeypatch.delenv("MINER_LOCAL_URL", raising=False)
    monkeypatch.delenv("MINER_LOCAL_MODEL", raising=False)
    monkeypatch.setenv("LOCAL_LLM_URL", "http://legacy.test/v1/chat/completions")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "legacy-model")

    def fake_post(url, payload, timeout):
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(providers, "_http_post_json", fake_post)
    out = providers.local_provider("s", "m", sleep=lambda s: None)
    assert out == "ok"


def test_local_provider_default_model_when_unset(monkeypatch):
    monkeypatch.setenv("MINER_LOCAL_URL", "http://local.test/v1")
    monkeypatch.delenv("MINER_LOCAL_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    seen = {}

    def fake_post(url, payload, timeout):
        seen["model"] = payload["model"]
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(providers, "_http_post_json", fake_post)
    providers.local_provider("s", "m", sleep=lambda s: None)
    assert seen["model"] == providers.DEFAULT_LOCAL_MODEL


def test_local_provider_honors_retry_wrapper(monkeypatch):
    """A transient 503 is retried via the SAME _with_retry wrapper, then succeeds."""
    monkeypatch.setenv("MINER_LOCAL_URL", "http://local.test/v1")
    attempts = {"n": 0}

    def flaky_post(url, payload, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr(providers, "_http_post_json", flaky_post)
    out = providers.local_provider("s", "m", sleep=lambda s: None)
    assert out == "[]"
    assert attempts["n"] == 3  # two 503s retried, third succeeds


def test_local_provider_non_retryable_http_raises(monkeypatch):
    monkeypatch.setenv("MINER_LOCAL_URL", "http://local.test/v1")

    def bad_post(url, payload, timeout):
        raise urllib.error.HTTPError(url, 400, "bad request", {}, None)

    monkeypatch.setattr(providers, "_http_post_json", bad_post)
    with pytest.raises(ProviderError):
        providers.local_provider("s", "m", sleep=lambda s: None)


def test_local_provider_exhausts_retries(monkeypatch):
    monkeypatch.setenv("MINER_LOCAL_URL", "http://local.test/v1")

    def always_500(url, payload, timeout):
        raise urllib.error.HTTPError(url, 500, "boom", {}, None)

    monkeypatch.setattr(providers, "_http_post_json", always_500)
    with pytest.raises(ProviderError):
        providers.local_provider("s", "m", sleep=lambda s: None)


def test_local_provider_empty_choices_returns_empty(monkeypatch):
    monkeypatch.setenv("MINER_LOCAL_URL", "http://local.test/v1")
    monkeypatch.setattr(providers, "_http_post_json",
                        lambda url, payload, timeout: {"choices": []})
    assert providers.local_provider("s", "m", sleep=lambda s: None) == ""
