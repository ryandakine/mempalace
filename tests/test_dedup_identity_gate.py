"""Plan §6: dedup never auto-updates user/feedback (identity gate)."""

from pathlib import Path

from mempalace.memory_miner.dedup import StoreIndex, classify
from mempalace.memory_miner.proposal import MemoryProposal


def _seed(tmp_path: Path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "feedback_skip_permissions.md").write_text(
        "---\nname: Skip permission prompts and just execute\n"
        "description: User prefers bypassing tool permissions and executing without interruption.\n"
        "type: feedback\n---\nDo not ask for permission before executing.\n"
    )
    (mem / "user_ryan_profile.md").write_text(
        "---\nname: Ryan profile\n"
        "description: Ryan's role, tech stack, products, and preferences.\n"
        "type: user\n---\nRyan is a developer.\n"
    )
    return mem


def _prop(name, ptype, desc, body):
    return MemoryProposal(name=name, type=ptype, description=desc, body=body,
                          source_session="s", source_chunk=0)


def test_feedback_near_duplicate_flags_review_not_update(tmp_path):
    idx = StoreIndex.from_dir(_seed(tmp_path))
    p = _prop("Skip permission prompts and just execute", "feedback",
              "User prefers bypassing tool permissions and executing without interruption.",
              "Never ask for permission; just execute.")
    v = classify(p, idx)
    assert v["action"] == "review"  # NOT 'update' — identity types are human-gated


def test_user_near_duplicate_flags_review_not_update(tmp_path):
    idx = StoreIndex.from_dir(_seed(tmp_path))
    p = _prop("Ryan profile", "user",
              "Ryan's role, tech stack, products, and preferences.",
              "Ryan is a senior developer with a broad stack.")
    v = classify(p, idx)
    assert v["action"] == "review"


def test_reference_high_overlap_still_updates(tmp_path):
    # contrast: a non-identity type at the same overlap DOES update.
    mem = _seed(tmp_path)
    (mem / "reference_thing.md").write_text(
        "---\nname: Some reference thing\n"
        "description: A durable external pointer about deployment topology details.\n"
        "type: reference\n---\nbody\n"
    )
    idx = StoreIndex.from_dir(mem)
    p = _prop("Some reference thing", "reference",
              "A durable external pointer about deployment topology details.",
              "Updated deployment topology details.")
    v = classify(p, idx)
    assert v["action"] == "update"
    assert v["enrich"] is True  # non-identity update enriches


def test_identity_near_dup_never_carries_enrich(tmp_path):
    # even at very high overlap, an identity type must NOT be flagged to enrich;
    # it routes to review with enrich False (human-gated forever).
    idx = StoreIndex.from_dir(_seed(tmp_path))
    p = _prop("Skip permission prompts and just execute", "feedback",
              "User prefers bypassing tool permissions and executing without interruption.",
              "Do not ask for permission before executing; just run the command.")
    v = classify(p, idx)
    assert v["action"] == "review"
    assert v["enrich"] is False
    assert v["target"] == "feedback_skip_permissions.md"  # match found, but gated
