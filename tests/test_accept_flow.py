"""Plan §6: accept promotes id → file + index + ledger; atomic. Integration."""

import json
from pathlib import Path

from mempalace.memory_miner.accept import accept
from mempalace.memory_miner.emit import emit, load_accepted_ids
from mempalace.memory_miner.proposal import MemoryProposal


def _setup(tmp_path: Path):
    out = tmp_path / "queue"
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "MEMORY.md").write_text("# Memory Index\n\n## Reference\n")
    props = [
        MemoryProposal(name="Docker deploy", type="reference",
                       description="Deploys via Docker Compose.",
                       body="Ryan deploys all projects via Docker Compose.",
                       source_session="sess-1", source_chunk=0, confidence=0.92),
        MemoryProposal(name="Low conf project", type="project",
                       description="Some project state.",
                       body="A project status fact.",
                       source_session="sess-1", source_chunk=1, confidence=0.40),
    ]
    emit(props, out)
    ledger = tmp_path / "accepted.jsonl"
    return out / "proposals.jsonl", mem, ledger, props


def test_accept_by_id_writes_file_index_ledger(tmp_path):
    queue, mem, ledger, props = _setup(tmp_path)
    results = accept(queue, ids=[props[0].id], memory_dir=mem, ledger_path=ledger)
    assert len(results) == 1
    fname = results[0]["file"]
    # memory file written with frontmatter
    written = (mem / fname).read_text()
    assert "type: reference" in written
    assert "Docker Compose" in written
    # MEMORY.md pointer added
    assert f"({fname})" in (mem / "MEMORY.md").read_text()
    # ledger appended
    assert props[0].id in load_accepted_ids(ledger)


def test_accept_above_threshold_with_type(tmp_path):
    queue, mem, ledger, props = _setup(tmp_path)
    # only the 0.92 reference qualifies for --above 0.85 --type reference
    results = accept(queue, above=0.85, type_filter="reference", memory_dir=mem, ledger_path=ledger)
    assert len(results) == 1
    assert results[0]["type"] == "reference"


def test_accept_is_idempotent_no_double_write(tmp_path):
    queue, mem, ledger, props = _setup(tmp_path)
    accept(queue, ids=[props[0].id], memory_dir=mem, ledger_path=ledger)
    # second accept of the same id → skipped (already in ledger)
    again = accept(queue, ids=[props[0].id], memory_dir=mem, ledger_path=ledger)
    assert again == []
    # only one memory file (besides MEMORY.md)
    files = [f.name for f in mem.glob("*.md") if f.name != "MEMORY.md"]
    assert len(files) == 1
    # MEMORY.md still has exactly one pointer for it
    text = (mem / "MEMORY.md").read_text()
    assert text.count(f"({files[0]})") == 1


def test_accept_nonexistent_id_noop(tmp_path):
    queue, mem, ledger, props = _setup(tmp_path)
    results = accept(queue, ids=["deadbeefdeadbeef"], memory_dir=mem, ledger_path=ledger)
    assert results == []


def test_accept_does_not_touch_unrelated_memory_files(tmp_path):
    queue, mem, ledger, props = _setup(tmp_path)
    (mem / "user_existing.md").write_text("---\nname: Existing\ndescription: d\ntype: user\n---\nb\n")
    before = (mem / "user_existing.md").read_text()
    accept(queue, ids=[props[0].id], memory_dir=mem, ledger_path=ledger)
    assert (mem / "user_existing.md").read_text() == before
