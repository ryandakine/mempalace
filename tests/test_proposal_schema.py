"""Plan §6: distill output schema-validated; malformed → reject."""

from mempalace.memory_miner.proposal import (
    MemoryProposal,
    proposal_id,
    validate_proposal,
)


def _good():
    return {
        "name": "ryan-prefers-pytest",
        "type": "feedback",
        "description": "Run pytest before commits.",
        "body": "Ryan treats a red test suite as a hard blocker.",
        "confidence": 0.9,
        "likely_duplicate_of": None,
    }


def test_valid_proposal_accepted():
    ok, reason = validate_proposal(_good())
    assert ok, reason


def test_missing_field_rejected():
    p = _good()
    del p["body"]
    ok, reason = validate_proposal(p)
    assert not ok
    assert "body" in reason


def test_empty_string_field_rejected():
    p = _good()
    p["name"] = "   "
    ok, _ = validate_proposal(p)
    assert not ok


def test_bad_type_rejected():
    p = _good()
    p["type"] = "milestone"  # not a store type
    ok, reason = validate_proposal(p)
    assert not ok
    assert "type" in reason


def test_confidence_out_of_range_rejected():
    p = _good()
    p["confidence"] = 1.7
    ok, _ = validate_proposal(p)
    assert not ok


def test_non_dict_rejected():
    ok, _ = validate_proposal("not a dict")
    assert not ok


def test_dup_must_be_string_or_null():
    p = _good()
    p["likely_duplicate_of"] = 12345
    ok, _ = validate_proposal(p)
    assert not ok


def test_proposal_id_stable():
    a = proposal_id("sess-1", 3, "the body text")
    b = proposal_id("sess-1", 3, "the body text")
    c = proposal_id("sess-1", 3, "different body")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_dataclass_autofills_id():
    p = MemoryProposal(
        name="x", type="reference", description="d", body="b", source_session="s", source_chunk=0
    )
    assert p.id == proposal_id("s", 0, "b")
