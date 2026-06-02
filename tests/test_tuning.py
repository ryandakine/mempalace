"""Task #6: decisions-ledger scaffold (tuning.py) + accept() decision hook.

Pure scaffold tests — no live calls, temp dirs only. Covers:
  - record + load roundtrip
  - decided_ids returns the right set
  - reject records a rejection (with meta passthrough)
  - accept() writes an 'accepted' decision
  - a logging failure never breaks a successful accept (unwritable ledger)
"""

import json
from pathlib import Path

import pytest

from mempalace.memory_miner.accept import accept
from mempalace.memory_miner.emit import emit
from mempalace.memory_miner.proposal import MemoryProposal
from mempalace.memory_miner.tuning import (
    decided_ids,
    default_decisions_path,
    load_decisions,
    record_decision,
    reject,
)


# --------------------------------------------------------------------------
# tuning.py unit tests
# --------------------------------------------------------------------------
def test_record_and_load_roundtrip(tmp_path):
    ledger = tmp_path / "decisions.jsonl"
    ok = record_decision("id-1", "accepted", type="reference", score=0.9,
                         meta={"why": "high conf"}, ledger_path=ledger)
    assert ok is True

    rows = load_decisions(ledger)
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == "id-1"
    assert r["action"] == "accepted"
    assert r["type"] == "reference"
    assert r["score"] == 0.9
    assert r["meta"] == {"why": "high conf"}
    # no wall-clock stamped when ts not provided (determinism)
    assert "ts" not in r


def test_record_omits_optional_fields_when_none(tmp_path):
    ledger = tmp_path / "decisions.jsonl"
    record_decision("id-x", "reviewed", ledger_path=ledger)
    r = load_decisions(ledger)[0]
    assert r == {"id": "id-x", "action": "reviewed"}


def test_record_includes_ts_when_passed(tmp_path):
    ledger = tmp_path / "decisions.jsonl"
    record_decision("id-ts", "accepted", ts="2026-06-01T00:00:00", ledger_path=ledger)
    r = load_decisions(ledger)[0]
    assert r["ts"] == "2026-06-01T00:00:00"


def test_record_is_append_only(tmp_path):
    ledger = tmp_path / "decisions.jsonl"
    record_decision("a", "accepted", ledger_path=ledger)
    record_decision("b", "rejected", ledger_path=ledger)
    record_decision("c", "merged", ledger_path=ledger)
    rows = load_decisions(ledger)
    assert [r["id"] for r in rows] == ["a", "b", "c"]


def test_record_empty_id_is_noop(tmp_path):
    ledger = tmp_path / "decisions.jsonl"
    assert record_decision("", "accepted", ledger_path=ledger) is False
    assert load_decisions(ledger) == []


def test_decided_ids_returns_right_set(tmp_path):
    ledger = tmp_path / "decisions.jsonl"
    record_decision("a", "accepted", ledger_path=ledger)
    record_decision("b", "rejected", ledger_path=ledger)
    record_decision("c", "merged", ledger_path=ledger)
    # a duplicate id for an already-decided proposal collapses in the set
    record_decision("a", "reviewed", ledger_path=ledger)
    assert decided_ids(ledger) == {"a", "b", "c"}


def test_decided_ids_empty_when_no_ledger(tmp_path):
    assert decided_ids(tmp_path / "nope.jsonl") == set()
    assert load_decisions(tmp_path / "nope.jsonl") == []


def test_load_tolerates_blank_and_malformed_lines(tmp_path):
    ledger = tmp_path / "decisions.jsonl"
    ledger.write_text(
        '{"id": "ok", "action": "accepted"}\n'
        "\n"
        "not json at all\n"
        '{"action": "rejected"}\n'   # missing id → skipped
        '{"id": "ok2", "action": "merged"}\n'
    )
    rows = load_decisions(ledger)
    assert {r["id"] for r in rows} == {"ok", "ok2"}
    assert decided_ids(ledger) == {"ok", "ok2"}


def test_reject_records_a_rejection(tmp_path):
    ledger = tmp_path / "decisions.jsonl"
    ok = reject("dead-id", ledger_path=ledger, type="project", score=0.2,
                reason="session-local noise")
    assert ok is True
    r = load_decisions(ledger)[0]
    assert r["id"] == "dead-id"
    assert r["action"] == "rejected"
    assert r["type"] == "project"
    assert r["score"] == 0.2
    # arbitrary kwargs land in meta
    assert r["meta"] == {"reason": "session-local noise"}


def test_reject_without_extra_meta(tmp_path):
    ledger = tmp_path / "decisions.jsonl"
    reject("plain-id", ledger_path=ledger)
    r = load_decisions(ledger)[0]
    assert r == {"id": "plain-id", "action": "rejected"}


def test_record_failure_is_best_effort(tmp_path):
    # point the ledger "path" at a directory → open() for append raises,
    # record_decision must swallow it and return False (never raise).
    bad = tmp_path / "is_a_dir"
    bad.mkdir()
    assert record_decision("x", "accepted", ledger_path=bad) is False


# --------------------------------------------------------------------------
# accept() decision hook
# --------------------------------------------------------------------------
def _accept_setup(tmp_path: Path):
    out = tmp_path / "queue"
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "MEMORY.md").write_text("# Memory Index\n\n## Reference\n")
    prop = MemoryProposal(
        name="Docker deploy", type="reference",
        description="Deploys via Docker Compose.",
        body="Ryan deploys all projects via Docker Compose.",
        source_session="sess-1", source_chunk=0, confidence=0.92,
    )
    emit([prop], out)
    ledger = tmp_path / "accepted.jsonl"
    return out / "proposals.jsonl", mem, ledger, prop


def test_accept_writes_accepted_decision(tmp_path):
    queue, mem, ledger, prop = _accept_setup(tmp_path)
    # the decisions ledger lives at the global default path; under the test
    # suite HOME is redirected to a temp dir (conftest), so this is isolated.
    decisions = default_decisions_path()
    # start from a clean slate for a deterministic assertion
    if decisions.exists():
        decisions.unlink()

    results = accept(queue, ids=[prop.id], memory_dir=mem, ledger_path=ledger)
    assert len(results) == 1
    assert results[0]["action"] == "create"

    rows = load_decisions(decisions)
    rec = next(r for r in rows if r["id"] == prop.id)
    assert rec["action"] == "accepted"
    assert rec["type"] == "reference"
    assert rec["score"] == 0.92
    assert prop.id in decided_ids(decisions)


def test_accept_merge_writes_merged_decision(tmp_path):
    out = tmp_path / "queue"
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "MEMORY.md").write_text(
        "# Memory Index\n\n## Reference\n- [grok](reference_grok.md) — old\n"
    )
    (mem / "reference_grok.md").write_text(
        "---\nname: Grok proxy\ndescription: d\ntype: reference\n---\n"
        "The grok proxy listens on 127.0.0.1:8765.\n"
    )
    prop = MemoryProposal(
        name="Grok proxy refreshed", type="reference",
        description="now streams", body="It also streams when stream=true.",
        source_session="s", source_chunk=0, confidence=0.95,
    )
    prop.dedup = {"action": "update", "target": "reference_grok.md",
                  "score": 0.8, "enrich": True}
    emit([prop], out)
    ledger = tmp_path / "accepted.jsonl"

    decisions = default_decisions_path()
    if decisions.exists():
        decisions.unlink()

    results = accept(out / "proposals.jsonl", ids=[prop.id], memory_dir=mem,
                     ledger_path=ledger)
    assert results[0]["action"] == "merge"

    rec = next(r for r in load_decisions(decisions) if r["id"] == prop.id)
    assert rec["action"] == "merged"


def test_accept_succeeds_when_decision_logging_fails(tmp_path, monkeypatch):
    # simulate an unwritable decisions ledger: record_decision raises.
    # accept() must still complete and return its normal result shape.
    queue, mem, ledger, prop = _accept_setup(tmp_path)

    import mempalace.memory_miner.accept as accept_mod

    def _boom(*args, **kwargs):
        raise OSError("decisions ledger is unwritable")

    monkeypatch.setattr(accept_mod, "record_decision", _boom)

    results = accept(queue, ids=[prop.id], memory_dir=mem, ledger_path=ledger)
    # accept still succeeded: file written, ledger appended, result returned.
    assert len(results) == 1
    assert results[0]["action"] == "create"
    assert (mem / results[0]["file"]).exists()
    from mempalace.memory_miner.emit import load_accepted_ids
    assert prop.id in load_accepted_ids(ledger)
