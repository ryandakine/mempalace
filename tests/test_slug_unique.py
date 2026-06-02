"""Plan §6: slug collision → -2/-3 at generation."""

from pathlib import Path

from mempalace.memory_miner.accept import memory_filename
from mempalace.memory_miner.dedup import StoreIndex


def test_store_index_unique_slug_appends_suffix(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "reference_my_fact.md").write_text(
        "---\nname: My fact\ndescription: d\ntype: reference\n---\nb\n"
    )
    idx = StoreIndex.from_dir(mem)
    # first request for an existing stem → suffixed
    s1 = idx.unique_slug("reference_my_fact")
    assert s1 == "reference_my_fact_2"
    # second collision → next suffix
    s2 = idx.unique_slug("reference_my_fact")
    assert s2 == "reference_my_fact_3"
    # a brand-new slug is returned unchanged
    assert idx.unique_slug("reference_brand_new") == "reference_brand_new"


def test_memory_filename_unique_against_disk():
    existing = {"reference_grok_proxy.md"}
    p1 = {"name": "Grok proxy", "type": "reference"}
    f1 = memory_filename(p1, existing)
    assert f1 == "reference_grok_proxy_2.md"  # already on disk → suffixed
    f2 = memory_filename(p1, existing)
    assert f2 == "reference_grok_proxy_3.md"


def test_memory_filename_prefixes_type():
    f = memory_filename({"name": "Brand new fact", "type": "project"}, set())
    assert f == "project_brand_new_fact.md"
