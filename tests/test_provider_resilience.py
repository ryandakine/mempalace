"""Plan §6: provider timeout / 429 / hard-fail → skip+log, no crash."""

import urllib.error

import pytest

from mempalace.memory_miner import providers
from mempalace.memory_miner.providers import ProviderError, _with_retry


class _Resp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_retry_succeeds_after_transient_429(monkeypatch):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError("u", 429, "rate limited", {}, None)
        return "ok"

    out = _with_retry(flaky, retries=3, sleep=lambda s: None)
    assert out == "ok"
    assert calls["n"] == 3


def test_non_retryable_http_raises_provider_error():
    def bad():
        raise urllib.error.HTTPError("u", 401, "unauthorized", {}, None)

    with pytest.raises(ProviderError):
        _with_retry(bad, retries=3, sleep=lambda s: None)


def test_exhausted_retries_raise_provider_error():
    def always_500():
        raise urllib.error.HTTPError("u", 500, "boom", {}, None)

    with pytest.raises(ProviderError):
        _with_retry(always_500, retries=2, sleep=lambda s: None)


def test_url_error_is_retried_then_fails():
    calls = {"n": 0}

    def down():
        calls["n"] += 1
        raise urllib.error.URLError("connection refused")

    with pytest.raises(ProviderError):
        _with_retry(down, retries=2, sleep=lambda s: None)
    assert calls["n"] == 2


def test_grok_provider_backoff_uses_injected_sleep(monkeypatch):
    """grok_provider survives transient 5xx and succeeds, without real sleeping."""
    attempts = {"n": 0}

    def fake_post(url, payload, timeout):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
        return {"content": "[]"}

    monkeypatch.setattr(providers, "_http_post_json", fake_post)
    out = providers.grok_provider("sys", "msg", sleep=lambda s: None)
    assert out == "[]"
    assert attempts["n"] == 2


def test_none_provider_raises():
    with pytest.raises(ProviderError):
        providers.none_provider("sys", "msg")
