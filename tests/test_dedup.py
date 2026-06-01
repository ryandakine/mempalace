"""Plan §6: dedup create/update/skip thresholds; FTS-absent fallback works."""

from pathlib import Path

from mempalace.memory_miner.dedup import StoreIndex, classify
from mempalace.memory_miner.proposal import MemoryProposal


def _seed_store(tmp_path: Path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "reference_grok_proxy.md").write_text(
        "---\nname: Ask Grok proxy endpoint\n"
        "description: Local FastAPI proxy forwarding to xAI Grok at port 8765.\n"
        "type: reference\n---\nThe grok proxy listens on 127.0.0.1:8765.\n"
    )
    (mem / "project_oregon_trail.md").write_text(
        "---\nname: Oregon Trail game\n"
        "description: AI-native Oregon Trail v3 primitive Kaplay rebuild.\n"
        "type: project\n---\nOregon Trail v3 shipped.\n"
    )
    (mem / "MEMORY.md").write_text("# Memory Index\n")
    return mem


def _prop(name, ptype, desc, body):
    return MemoryProposal(name=name, type=ptype, description=desc, body=body,
                          source_session="s", source_chunk=0)


def test_high_overlap_reference_becomes_update(tmp_path):
    idx = StoreIndex.from_dir(_seed_store(tmp_path))
    p = _prop("Ask Grok proxy endpoint", "reference",
              "Local FastAPI proxy forwarding to xAI Grok at port 8765.",
              "The grok proxy forwards to xAI Grok on port 8765.")
    v = classify(p, idx)
    assert v["action"] == "update"
    assert v["target"] == "reference_grok_proxy.md"


def test_no_overlap_becomes_create_with_unique_slug(tmp_path):
    idx = StoreIndex.from_dir(_seed_store(tmp_path))
    p = _prop("Stripe webhook signing secret rotation", "reference",
              "Rotate Stripe webhook signing secret quarterly.",
              "Stripe webhook signing secrets rotate every quarter.")
    v = classify(p, idx)
    assert v["action"] == "create"
    assert v["target"] is None
    # a fresh, non-colliding slug was reserved
    assert v["slug"] == "stripe_webhook_signing_secret_rotation"
    assert v["slug"] not in {"reference_grok_proxy", "project_oregon_trail"}


def test_mid_overlap_flags_review(tmp_path):
    idx = StoreIndex.from_dir(_seed_store(tmp_path))
    # shares most tokens with the existing grok-proxy entry but is not an exact
    # name/slug match → lands in the MID band (>=0.42, <0.62) → review.
    p = _prop("Grok proxy endpoint", "reference",
              "Local FastAPI proxy forwarding to xAI Grok.",
              "The grok proxy forwards requests.")
    v = classify(p, idx)
    assert v["action"] == "review"
    assert 0.42 <= v["score"] < 0.62


def test_works_without_fts_no_embed_scorer(tmp_path):
    # No embed_scorer passed → pure keyword path still classifies.
    idx = StoreIndex.from_dir(_seed_store(tmp_path))
    p = _prop("Oregon Trail game", "project",
              "AI-native Oregon Trail v3 primitive Kaplay rebuild.",
              "Oregon Trail v3 primitive Kaplay rebuild shipped.")
    v = classify(p, idx, embed_scorer=None)
    assert v["action"] == "update"
    assert v["target"] == "project_oregon_trail.md"


def test_embed_scorer_can_boost(tmp_path):
    idx = StoreIndex.from_dir(_seed_store(tmp_path))
    p = _prop("totally unrelated topic xyz", "reference", "nothing alike", "nothing alike at all")
    # without booster → create
    assert classify(p.__class__(**{**p.to_dict()}), idx)["action"] == "create"
    # with a booster that returns high similarity → update
    p2 = _prop("totally unrelated topic xyz", "reference", "nothing alike", "nothing alike at all")
    v = classify(p2, idx, embed_scorer=lambda prop, entry: 0.99)
    assert v["action"] == "update"


def test_index_lines_render(tmp_path):
    idx = StoreIndex.from_dir(_seed_store(tmp_path))
    lines = idx.index_lines()
    assert "Ask Grok proxy endpoint" in lines
    assert "Oregon Trail game" in lines


def test_empty_store_dir(tmp_path):
    idx = StoreIndex.from_dir(tmp_path / "does-not-exist")
    p = _prop("new fact", "reference", "d", "b")
    v = classify(p, idx)
    assert v["action"] == "create"
