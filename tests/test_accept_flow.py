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


# --------------------------------------------------------------------------
# Merge / enrich path: a proposal whose dedup verdict is update+enrich must
# ENRICH the existing target file instead of creating a duplicate.
# --------------------------------------------------------------------------
def _enrich_setup(tmp_path: Path):
    out = tmp_path / "queue"
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "MEMORY.md").write_text(
        "# Memory Index\n\n## Reference\n- [grok proxy](reference_grok_proxy.md) — old desc\n"
    )
    (mem / "reference_grok_proxy.md").write_text(
        "---\nname: Ask Grok proxy endpoint\n"
        "description: Local FastAPI proxy forwarding to xAI Grok at port 8765.\n"
        "type: reference\n---\nThe grok proxy listens on 127.0.0.1:8765.\n"
    )
    prop = MemoryProposal(
        name="Ask Grok proxy endpoint refreshed", type="reference",
        description="Local FastAPI proxy to xAI Grok; now supports streaming.",
        body="The grok proxy also streams responses when stream=true is set.",
        source_session="sess-9", source_chunk=0, confidence=0.95,
    )
    # simulate the dedup verdict the classifier would have produced
    prop.dedup = {"action": "update", "target": "reference_grok_proxy.md",
                  "score": 0.8, "enrich": True}
    emit([prop], out)
    ledger = tmp_path / "accepted.jsonl"
    return out / "proposals.jsonl", mem, ledger, prop


def test_accept_merge_enriches_existing_file_no_duplicate(tmp_path):
    queue, mem, ledger, prop = _enrich_setup(tmp_path)
    results = accept(queue, ids=[prop.id], memory_dir=mem, ledger_path=ledger)
    assert len(results) == 1
    assert results[0]["action"] == "merge"
    assert results[0]["file"] == "reference_grok_proxy.md"

    # NO new file created — only the existing one + MEMORY.md
    files = sorted(f.name for f in mem.glob("*.md") if f.name != "MEMORY.md")
    assert files == ["reference_grok_proxy.md"]

    # the new info was appended onto the existing body
    text = (mem / "reference_grok_proxy.md").read_text()
    assert "127.0.0.1:8765" in text          # original body preserved
    assert "streams responses" in text        # delta appended
    # description refreshed to the newer proposal's description
    assert "now supports streaming" in text

    # MEMORY.md still has exactly one pointer to the file (replace-in-place)
    idx_text = (mem / "MEMORY.md").read_text()
    assert idx_text.count("(reference_grok_proxy.md)") == 1


def test_accept_merge_is_idempotent(tmp_path):
    queue, mem, ledger, prop = _enrich_setup(tmp_path)
    accept(queue, ids=[prop.id], memory_dir=mem, ledger_path=ledger)
    body_after_first = (mem / "reference_grok_proxy.md").read_text()
    # second accept of same id → skipped via ledger; body unchanged
    again = accept(queue, ids=[prop.id], memory_dir=mem, ledger_path=ledger)
    assert again == []
    assert (mem / "reference_grok_proxy.md").read_text() == body_after_first


def test_accept_merge_target_missing_falls_back_to_create(tmp_path):
    # if the dedup target no longer exists on disk, enrich must not crash — it
    # falls back to creating a fresh file.
    queue, mem, ledger, prop = _enrich_setup(tmp_path)
    (mem / "reference_grok_proxy.md").unlink()
    results = accept(queue, ids=[prop.id], memory_dir=mem, ledger_path=ledger)
    assert len(results) == 1
    assert results[0]["action"] == "create"
    assert (mem / results[0]["file"]).exists()


def test_accept_identity_update_does_not_merge(tmp_path):
    # an identity-type proposal that somehow carries an update verdict must NOT
    # enrich an existing identity file — it creates a fresh file for a human to
    # reconcile (identity types are human-gated forever).
    out = tmp_path / "queue"
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "MEMORY.md").write_text("# Memory Index\n\n## Feedback\n")
    (mem / "feedback_skip.md").write_text(
        "---\nname: Skip prompts\ndescription: just execute.\ntype: feedback\n---\noriginal feedback\n"
    )
    prop = MemoryProposal(
        name="Skip prompts", type="feedback",
        description="just execute, never ask.",
        body="Always skip permission prompts.",
        source_session="s", source_chunk=0, confidence=0.95,
    )
    prop.dedup = {"action": "update", "target": "feedback_skip.md",
                  "score": 0.9, "enrich": True}  # even if mis-set, must be ignored
    emit([prop], out)
    ledger = tmp_path / "accepted.jsonl"
    results = accept(out / "proposals.jsonl", ids=[prop.id], memory_dir=mem, ledger_path=ledger)
    assert results[0]["action"] == "create"
    # original feedback file untouched
    assert (mem / "feedback_skip.md").read_text() == \
        "---\nname: Skip prompts\ndescription: just execute.\ntype: feedback\n---\noriginal feedback\n"
