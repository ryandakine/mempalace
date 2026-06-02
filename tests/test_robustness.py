"""Plan §6: empty/binary/None transcript skipped, logged, no crash."""

from pathlib import Path

from mempalace.memory_miner.distill import distill, extract_segments


def _provider_never_called(system, message, *, timeout=180.0, max_tokens=8000):
    raise AssertionError("provider should not be called for unusable transcripts")


def test_empty_file_skipped(tmp_path):
    t = tmp_path / "empty.jsonl"
    t.write_text("")
    out = distill(t, session_id="empty", provider=_provider_never_called)
    assert out == []


def test_binary_file_skipped(tmp_path):
    t = tmp_path / "binary.jsonl"
    t.write_bytes(bytes(range(256)) * 4)
    out = distill(t, session_id="binary", provider=_provider_never_called)
    assert out == []


def test_nonexistent_file_skipped(tmp_path):
    out = distill(tmp_path / "nope.jsonl", session_id="nope", provider=_provider_never_called)
    assert out == []


def test_malformed_json_lines_skipped(tmp_path):
    t = tmp_path / "bad.jsonl"
    t.write_text("not json\n{broken\n{}\n")
    out = distill(t, session_id="bad", provider=_provider_never_called)
    assert out == []


def test_extract_segments_on_garbage(tmp_path):
    t = tmp_path / "garbage.jsonl"
    t.write_text("\x00\x01\x02 garbage not json at all\n")
    # no exception, returns empty
    assert extract_segments(t) == []
