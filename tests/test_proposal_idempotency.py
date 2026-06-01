"""Plan §6: proposal id stable; re-run dedups proposals.jsonl (no append dupes)."""

from pathlib import Path

from mempalace.memory_miner.emit import emit, load_accepted_ids, load_existing_ids
from mempalace.memory_miner.proposal import MemoryProposal


def _props():
    return [
        MemoryProposal(name="a", type="reference", description="da", body="ba",
                       source_session="s1", source_chunk=0, confidence=0.7),
        MemoryProposal(name="b", type="project", description="db", body="bb",
                       source_session="s1", source_chunk=1, confidence=0.6),
    ]


def test_rerun_does_not_duplicate(tmp_path):
    out = tmp_path / "queue"
    first = emit(_props(), out)
    assert len(first) == 2
    # second emit of the SAME proposals → nothing new appended
    second = emit(_props(), out)
    assert second == []

    ids = load_existing_ids(out / "proposals.jsonl")
    assert len(ids) == 2  # still exactly two lines


def test_accepted_ids_are_skipped(tmp_path):
    out = tmp_path / "queue"
    ledger = tmp_path / "accepted.jsonl"
    props = _props()
    emit(props, out)  # queue them

    # mark the first as accepted in the ledger
    import json
    with open(ledger, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": props[0].id, "file": "x.md"}) + "\n")

    # re-emitting with the ledger known → already-queued still skipped, and a NEW
    # proposal that happens to be accepted is also skipped.
    new_prop = MemoryProposal(name="c", type="reference", description="dc", body="bc",
                              source_session="s1", source_chunk=2)
    # pretend c was already accepted
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": new_prop.id, "file": "c.md"}) + "\n")

    written = emit([new_prop], out, accepted_ledger=ledger)
    assert written == []  # skipped because it's in the accepted ledger
    assert new_prop.id in load_accepted_ids(ledger)


def test_proposals_md_regenerated(tmp_path):
    out = tmp_path / "queue"
    emit(_props(), out)
    md = (out / "proposals.md").read_text()
    assert "Proposal Queue" in md
    assert "da" in md or "ba" in md
