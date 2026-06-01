"""Plan §4.5 / type-accuracy bug — sharpen 4-type classification in distill.

The live dry-run mistyped "Ryan routes heavy work through Grok when budget is
low" as `user` when it is really `feedback` (how to work with the user). This
suite locks in the fix on three layers:

  1. nudge_type(): the deterministic post-classification correction in distill.py
     (the only seam where the *final* type is decidable without a live LLM, since
     the type otherwise comes verbatim from the provider output).
  2. distill() end-to-end with a MOCKED provider returning representative proposal
     bodies — one per store type — asserting the type that actually lands.
  3. The DISTILL_SYSTEM prompt itself carries the disambiguation rules and a
     synthetic few-shot example for each of the 4 types.

No live provider calls anywhere.
"""

import json
from pathlib import Path

import pytest

from mempalace.memory_miner.distill import DISTILL_SYSTEM, distill, nudge_type


# --------------------------------------------------------------------------- #
# Layer 1: the deterministic nudge in isolation
# --------------------------------------------------------------------------- #

def test_nudge_routes_through_grok_user_to_feedback():
    """The exact bug: a routing rule mistyped `user` is corrected to `feedback`."""
    body = "Ryan routes heavy work through Grok when the budget is low."
    assert nudge_type("user", body) == "feedback"


@pytest.mark.parametrize("body", [
    "Ryan prefers the cheaper model and always reaches for it before the premium one.",
    "Never push directly to main; the user treats that as a hard rule.",
    "When the budget is tight, default to the local model instead of the API.",
    "Don't add comments to code you didn't change.",
])
def test_nudge_directives_become_feedback(body):
    assert nudge_type("user", body) == "feedback"


@pytest.mark.parametrize("body", [
    "Ryan is a staff backend engineer who builds betting and security products.",
    "The user works at an industrial recycling company and owns five products.",
    "The user's name is Brenny and his email is on file.",
    "The user develops on a 48GB Linux workstation with one free M.2 slot.",
])
def test_nudge_leaves_pure_identity_as_user(body):
    """Identity facts with no how-to-work directive stay `user`."""
    assert nudge_type("user", body) == "user"


@pytest.mark.parametrize("ptype,body", [
    ("feedback", "Ryan routes heavy work through Grok when budget is low."),
    ("project", "The billing rewrite is blocked on a pending schema migration."),
    ("reference", "Logs live in Grafana Loki; query by the service label."),
])
def test_nudge_never_touches_non_user_types(ptype, body):
    """The nudge only ever rewrites `user`; every other type passes through."""
    assert nudge_type(ptype, body) == ptype


def test_nudge_handles_empty_and_unknown_safely():
    assert nudge_type("user", "") == "user"
    assert nudge_type("user", None) == "user"  # type: ignore[arg-type]
    assert nudge_type("reference", "") == "reference"


# --------------------------------------------------------------------------- #
# Layer 2: end-to-end distill() with a mocked provider, one fact per type
# --------------------------------------------------------------------------- #

def _transcript(tmp_path: Path) -> Path:
    t = tmp_path / "sess.jsonl"
    rows = [
        {"message": {"role": "user",
                     "content": "Route heavy work through Grok when budget is low."}},
        {"message": {"role": "assistant", "content": "Understood."}},
    ]
    with open(t, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return t


def _by_name(proposals):
    return {p.name: p for p in proposals}


def test_distill_assigns_correct_types_per_intent(tmp_path):
    """A mocked provider returns one body per intent; assert the type that lands.

    The budget/routing fact is deliberately returned as `user` (the live bug) to
    prove the nudge promotes it to `feedback` end-to-end. The other three are
    returned with their correct type and must be preserved untouched.
    """
    def provider(system, message, *, timeout=180.0, max_tokens=8000):
        return json.dumps([
            # the bug: provider mistypes a routing RULE as user → nudge → feedback
            {"name": "grok-budget-routing", "type": "user",
             "description": "Routing rule under budget pressure.",
             "body": "Ryan routes heavy work through Grok when the budget is low.",
             "confidence": 0.9, "likely_duplicate_of": None},
            # identity → user (stays user)
            {"name": "ryan-identity", "type": "user",
             "description": "Who the user is.",
             "body": "Ryan is a staff backend engineer who builds betting and security products.",
             "confidence": 0.9, "likely_duplicate_of": None},
            # ongoing-work status → project (stays project)
            {"name": "billing-rewrite-status", "type": "project",
             "description": "In-flight work status.",
             "body": "The billing rewrite is blocked on a pending schema migration and ship is paused.",
             "confidence": 0.8, "likely_duplicate_of": None},
            # tool/path pointer → reference (stays reference)
            {"name": "loki-logs", "type": "reference",
             "description": "Where logs live.",
             "body": "Production logs live in Grafana Loki; query by the service label.",
             "confidence": 0.8, "likely_duplicate_of": None},
        ])

    out = distill(_transcript(tmp_path), session_id="sess", provider=provider)
    by_name = _by_name(out)

    # the headline assertion: the routing rule is now feedback, not user
    assert by_name["grok-budget-routing"].type == "feedback"
    # the three correctly-typed facts are preserved
    assert by_name["ryan-identity"].type == "user"
    assert by_name["billing-rewrite-status"].type == "project"
    assert by_name["loki-logs"].type == "reference"


def test_distill_does_not_over_correct_identity(tmp_path):
    """A pure identity fact the provider already typed `user` must remain `user`."""
    def provider(system, message, *, timeout=180.0, max_tokens=8000):
        return json.dumps([
            {"name": "ryan-role", "type": "user",
             "description": "Identity.",
             "body": "The user works at an industrial recycling company and owns five products.",
             "confidence": 0.9, "likely_duplicate_of": None},
        ])

    out = distill(_transcript(tmp_path), session_id="sess", provider=provider)
    assert len(out) == 1
    assert out[0].type == "user"


# --------------------------------------------------------------------------- #
# Layer 3: the prompt carries the sharpened guidance
# --------------------------------------------------------------------------- #

def test_prompt_contains_disambiguation_rules():
    p = DISTILL_SYSTEM.lower()
    assert "disambiguation" in p
    # the core rule: routes/prefers/always/never → feedback, not user
    assert "routes/prefers/always/never" in p
    assert "not user" in p
    # the literal bug example is encoded as guidance
    assert "routes heavy work through grok" in p


def test_prompt_has_a_synthetic_example_for_each_type():
    p = DISTILL_SYSTEM
    # the few-shot block labels one example per store type
    for label in ("user:", "feedback:", "project:", "reference:"):
        assert label in p, f"missing few-shot example labelled {label!r}"
    # examples are synthetic (no real session facts copied verbatim)
    assert "SHAPE ONLY" in p
