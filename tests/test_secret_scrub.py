"""Plan §6: secret_scrub redacts planted sk-/JWT/.env before provider+disk."""

from mempalace.memory_miner.scrub import secret_scrub


def test_redacts_anthropic_key():
    text = "my key is sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA and that's it"
    out, n = secret_scrub(text)
    assert "sk-ant-" not in out
    assert "[REDACTED:anthropic]" in out
    assert n >= 1


def test_redacts_openai_key():
    out, n = secret_scrub("export OPENAI=sk-abcdefghijklmnopqrstuvwxyz123456")
    assert "sk-abcdefghij" not in out
    assert n >= 1


def test_redacts_jwt():
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkw.dozjgNryP4J3jVmNHl0w"
    out, n = secret_scrub(f"token={jwt}")
    assert jwt not in out
    assert n >= 1


def test_redacts_env_assignment():
    # `password = ...` / `api_key: ...` / `token=...` shapes are caught by the
    # deterministic `assign` pattern (word-boundary keyword + value).
    out, n = secret_scrub('password = "hunter2hunter2hunter2"')
    assert "hunter2hunter2" not in out
    assert n >= 1
    out2, n2 = secret_scrub("API_KEY=abcdef0123456789xyz")
    assert "abcdef0123456789xyz" not in out2
    assert n2 >= 1


def test_redacts_aws_and_github():
    out, n = secret_scrub("AKIAIOSFODNN7EXAMPLE and ghp_0123456789abcdefghijklmnopqrstuvwxyz")
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "ghp_0123456789" not in out
    assert n >= 2


def test_clean_text_untouched():
    clean = "Ryan prefers running the full test suite before every commit."
    out, n = secret_scrub(clean)
    assert out == clean
    assert n == 0


def test_idempotent_on_redacted():
    once, _ = secret_scrub("key sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA")
    twice, n2 = secret_scrub(once)
    assert twice == once
    assert n2 == 0


def test_empty():
    assert secret_scrub("") == ("", 0)
