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
    from mempalace.memory_miner.dedup import HIGH, MID
    idx = StoreIndex.from_dir(_seed_store(tmp_path))
    # shares the "xAI Grok" topic with the existing grok-proxy entry but is a
    # distinct fact (not the proxy) → lands in the MID band [MID, HIGH) → review.
    p = _prop("xAI Grok model", "reference",
              "Grok is xAI's model.",
              "xAI makes the Grok model.")
    v = classify(p, idx)
    assert v["action"] == "review"
    assert MID <= v["score"] < HIGH


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


# --------------------------------------------------------------------------
# Recall regression: the proven-missed ask-claude-bridge-live vs ask-claude-bridge
# case. Under the old name/description-only lexical scorer it scored 0.31 and was
# wrongly classified `create`. With body-aware semantic similarity it must MATCH.
# --------------------------------------------------------------------------
def _seed_bridge(tmp_path: Path):
    mem = tmp_path / "memory"
    mem.mkdir()
    # mirrors the real reference_ask_claude_bridge.md (nested metadata frontmatter)
    (mem / "reference_ask_claude_bridge.md").write_text(
        "---\n"
        "name: ask-claude-bridge\n"
        "description: ask-claude is the reverse of ask-grok — lets Grok/any agent "
        "get a Claude review off-subscription\n"
        "metadata:\n"
        "  node_type: memory\n"
        "  type: reference\n"
        "---\n\n"
        "`/home/ryan/bin/ask-claude` is the mirror of ask-grok: it lets Grok or any "
        "tool-enabled agent get a Claude second opinion. Hits the Anthropic Messages "
        "API directly, default model claude-opus-4-8. Billing is pay-as-you-go "
        "Anthropic API, separate from the Claude Code Max subscription.\n"
    )
    return mem


def test_ask_claude_bridge_live_matches_existing(tmp_path):
    idx = StoreIndex.from_dir(_seed_bridge(tmp_path))
    p = _prop(
        "ask-claude-bridge-live", "reference",
        "ask-claude bridge lets Grok get a Claude review off the subscription, "
        "pay-as-you-go Anthropic API",
        "ask-claude is the mirror of ask-grok: it lets Grok or any tool-enabled "
        "agent get a Claude second opinion via the Anthropic Messages API, billed "
        "pay-as-you-go separate from the Claude Code subscription.",
    )
    best, score = idx.best_match(p)
    assert best is not None
    assert best["file"] == "reference_ask_claude_bridge.md"
    v = classify(p, idx)
    assert v["action"] == "update", f"expected update, got {v} (score={score})"
    assert v["target"] == "reference_ask_claude_bridge.md"
    assert v["enrich"] is True  # non-identity update → merge/enrich


def test_body_indexing_catches_dup_that_name_desc_miss(tmp_path):
    # name + description deliberately share almost NO tokens with the stored
    # entry; the duplicate signal lives entirely in the body.
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "reference_tunnel.md").write_text(
        "---\nname: Cloudflare tunnel inventory\n"
        "description: which named tunnel serves which hostname.\n"
        "type: reference\n---\n"
        "The main cloudflared tunnel is named congressional-intel and uses "
        "REMOTE config mode, load-balancing every hostname across connectors.\n"
    )
    idx = StoreIndex.from_dir(mem)
    p = _prop(
        "Shared connector load balancing surprise", "reference",
        "an operational gotcha about hostnames",
        "The main cloudflared tunnel is named congressional-intel and uses REMOTE "
        "config mode, load-balancing every hostname across connectors.",
    )
    # name/description alone barely overlap; body makes this a clear duplicate.
    best, score = idx.best_match(p)
    assert best["file"] == "reference_tunnel.md"
    assert classify(p, idx)["action"] == "update"


def test_nested_metadata_type_is_parsed(tmp_path):
    # real store files nest `type:` under a `metadata:` block, indented.
    idx = StoreIndex.from_dir(_seed_bridge(tmp_path))
    assert len(idx.entries) == 1
    assert idx.entries[0]["type"] == "reference"
    assert idx.entries[0]["name"] == "ask-claude-bridge"
