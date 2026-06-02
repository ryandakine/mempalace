"""discover_transcripts must skip subagent transcripts and workflow journals —
they're implementation chatter, not durable user sessions (first-real-run finding)."""

from mempalace.memory_miner.watermark import _is_subtranscript, discover_transcripts


def test_is_subtranscript_classification(tmp_path):
    root = tmp_path / "-home-ryan"
    assert _is_subtranscript(root / "abc-123.jsonl") is False
    assert _is_subtranscript(root / "abc-123" / "subagents" / "agent-xyz.jsonl") is True
    assert _is_subtranscript(root / "x" / "subagents" / "workflows" / "wf_1" / "journal.jsonl") is True
    assert _is_subtranscript(root / "agent-top-level.jsonl") is True  # agent- prefix anywhere


def test_discover_excludes_subtranscripts(tmp_path):
    root = tmp_path / "-home-ryan"
    (root).mkdir(parents=True)
    real1 = root / "11111111-aaaa.jsonl"
    real2 = root / "22222222-bbbb.jsonl"
    sub = root / "11111111-aaaa" / "subagents"
    sub.mkdir(parents=True)
    agent = sub / "agent-deadbeef.jsonl"
    journal = sub / "workflows" / "wf_x"
    journal.mkdir(parents=True)
    jfile = journal / "journal.jsonl"
    for f in (real1, real2, agent, jfile):
        f.write_text("{}\n")

    found = discover_transcripts([tmp_path])
    names = {p.name for p in found}
    assert names == {"11111111-aaaa.jsonl", "22222222-bbbb.jsonl"}
    assert "agent-deadbeef.jsonl" not in names
    assert "journal.jsonl" not in names


def test_filter_applies_with_newest_first(tmp_path):
    root = tmp_path / "-home-ryan"
    root.mkdir(parents=True)
    real = root / "33333333-cccc.jsonl"
    real.write_text("{}\n")
    sub = root / "s" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-z.jsonl").write_text("{}\n")

    found = discover_transcripts([tmp_path], newest_first=True)
    assert [p.name for p in found] == ["33333333-cccc.jsonl"]
