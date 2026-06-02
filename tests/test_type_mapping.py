"""Plan §6: type mapping table correct for all 5→4."""

from mempalace.memory_miner.proposal import STORE_TYPES, map_regex_type


def test_all_five_regex_types_map_to_store_types():
    for rt in ("preference", "decision", "milestone", "problem", "emotional"):
        st = map_regex_type(rt)
        assert st in STORE_TYPES, f"{rt} -> {st}"


def test_specific_mappings():
    assert map_regex_type("preference") == "feedback"
    assert map_regex_type("milestone") == "project"
    assert map_regex_type("decision") == "project"
    assert map_regex_type("problem") == "project"
    assert map_regex_type("emotional") == "user"


def test_unknown_and_none_fall_back_to_reference():
    assert map_regex_type("totally-unknown") == "reference"
    assert map_regex_type(None) == "reference"
    assert map_regex_type("") == "reference"


def test_case_insensitive():
    assert map_regex_type("PREFERENCE") == "feedback"
    assert map_regex_type("  Milestone ") == "project"
