"""Plan §6: frontmatter valid; MEMORY.md pointer idempotent (replace-in-place)."""

from pathlib import Path

from mempalace.memory_miner.accept import render_memory_md, update_memory_index


def test_render_memory_md_frontmatter():
    md = render_memory_md({
        "name": "Ryan deploys via Docker Compose",
        "description": "All projects deploy via Docker Compose + Cloudflare tunnels.",
        "type": "reference",
        "body": "Ryan deploys all projects via Docker Compose with Cloudflare tunnels.",
        "source_session": "abc-123",
    })
    assert md.startswith("---\n")
    assert "name: Ryan deploys via Docker Compose" in md
    assert "description: All projects deploy" in md
    assert "type: reference" in md
    assert "originSessionId: abc-123" in md
    # frontmatter closes and body follows
    assert md.count("---") == 2
    assert md.rstrip().endswith("Cloudflare tunnels.")


def test_index_appends_into_right_section(tmp_path):
    idx = tmp_path / "MEMORY.md"
    idx.write_text("# Memory Index\n\n## User\n- [existing](user_x.md) — x\n\n## Reference\n")
    update_memory_index(idx, "reference_new.md", "New reference fact",
                        "A durable pointer.", "reference")
    text = idx.read_text()
    # appended under ## Reference, not ## User
    ref_section = text.split("## Reference", 1)[1]
    assert "reference_new.md" in ref_section
    assert "New reference fact".lower().replace(" ", " ") in text.lower() or "new reference fact" in text.lower()


def test_index_creates_missing_section(tmp_path):
    idx = tmp_path / "MEMORY.md"
    idx.write_text("# Memory Index\n\n## User\n- [a](user_a.md) — a\n")
    update_memory_index(idx, "project_p.md", "Project thing", "desc", "project")
    text = idx.read_text()
    assert "## Project" in text
    assert "project_p.md" in text


def test_index_replace_in_place_no_duplicate(tmp_path):
    idx = tmp_path / "MEMORY.md"
    idx.write_text("# Memory Index\n\n## Reference\n- [old label](reference_x.md) — old desc\n")
    # update the SAME filename's pointer
    update_memory_index(idx, "reference_x.md", "New label", "new desc", "reference")
    text = idx.read_text()
    # exactly one pointer line for reference_x.md
    assert text.count("(reference_x.md)") == 1
    assert "new desc" in text
    assert "old desc" not in text


def test_index_idempotent_repeated_calls(tmp_path):
    idx = tmp_path / "MEMORY.md"
    idx.write_text("# Memory Index\n\n## Reference\n")
    for _ in range(3):
        update_memory_index(idx, "reference_y.md", "Y fact", "y desc", "reference")
    text = idx.read_text()
    assert text.count("(reference_y.md)") == 1


def test_index_created_when_absent(tmp_path):
    idx = tmp_path / "MEMORY.md"  # does not exist yet
    update_memory_index(idx, "reference_z.md", "Z fact", "z desc", "reference")
    assert idx.exists()
    assert "(reference_z.md)" in idx.read_text()
