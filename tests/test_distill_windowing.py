"""Large transcripts must be windowed, not dropped — first real run showed 40% of
recent sessions exceeded the provider's 200k char limit and hard-failed (HTTP 422)."""

import json

import mempalace.memory_miner.distill as D


def _transcript(tmp_path):
    p = tmp_path / "sess.jsonl"
    p.write_text("\n".join([
        json.dumps({"message": {"role": "user", "content": "hello about grafana loki for logs"}}),
        json.dumps({"message": {"role": "assistant", "content": "billing rewrite is blocked"}}),
    ]))
    return p


def test_window_splits_on_budget():
    text = "\n\n".join(["x" * 100 for _ in range(5)])
    w = D._window(text, 250)
    assert len(w) >= 2 and all(len(x) <= 250 for x in w)
    assert D._window("short", 1000) == ["short"]          # under budget → single
    w2 = D._window("y" * 1000, 300)                        # oversized single paragraph
    assert len(w2) == 4 and all(len(x) <= 300 for x in w2)


def test_distill_collects_and_dedups_across_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_window", lambda text, budget: ["WIN_A", "WIN_B"])
    calls = []

    def provider(system, message, *, timeout, max_tokens):
        calls.append(message)
        if "WIN_A" in message:
            return json.dumps([{"name": "fact-a", "type": "reference",
                                "description": "d", "body": "loki for logs"}])
        return json.dumps([
            {"name": "fact-a", "type": "reference", "description": "d", "body": "loki for logs"},
            {"name": "fact-b", "type": "project", "description": "d2", "body": "billing blocked"},
        ])

    out = D.distill(_transcript(tmp_path), session_id="sess",
                    provider=provider, existing_index="", today="2026-06-01")
    assert len(calls) == 2                          # one provider call per window
    assert sorted(o.name for o in out) == ["fact-a", "fact-b"]   # cross-window dup collapsed
