"""Provenance + staleness (Task #5): source_excerpt provenance, time_sensitive
heuristic, render into proposals.md, and schema/round-trip stability.

All provider calls are mocked; all writes go to pytest tmp dirs.
"""

import json
from pathlib import Path

from mempalace.memory_miner.distill import distill, is_time_sensitive
from mempalace.memory_miner.emit import emit
from mempalace.memory_miner.proposal import MemoryProposal, validate_proposal


# --------------------------------------------------------------------- helpers
def _write_transcript(tmp_path: Path, rows) -> Path:
    t = tmp_path / "sess.jsonl"
    with open(t, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return t


def _provider_returning(items):
    """A mocked provider that always replies with the given proposal list."""
    def fake_provider(system, message, *, timeout=180.0, max_tokens=8000):
        return json.dumps(items)
    return fake_provider


# ------------------------------------------------- source_excerpt is populated
def test_source_excerpt_drawn_from_transcript(tmp_path):
    # The distinctive transcript line the fact should be traced back to.
    origin = "I always run the full pytest suite before committing any change."
    rows = [
        {"message": {"role": "user", "content": origin}},
        {"message": {"role": "assistant", "content": "Understood, I'll do that."}},
    ]
    t = _write_transcript(tmp_path, rows)

    provider = _provider_returning([
        {"name": "run-pytest", "type": "feedback",
         "description": "Run pytest before commits.",
         "body": "The user runs the full pytest suite before committing.",
         "confidence": 0.9, "likely_duplicate_of": None},
    ])

    out = distill(t, session_id="sess", provider=provider)
    assert len(out) == 1
    exc = out[0].source_excerpt
    assert exc, "source_excerpt must be populated"
    # It is the ORIGINAL transcript passage, not the model's rewritten body.
    assert "pytest" in exc.lower()
    assert "before committing" in exc.lower()
    assert exc != out[0].body


def test_source_excerpt_is_secret_scrubbed(tmp_path):
    # A secret in the transcript must not survive into the excerpt.
    secret = "sk-ant-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    rows = [
        {"message": {"role": "user",
                     "content": f"Deploy uses the anthropic key {secret} for the worker."}},
    ]
    t = _write_transcript(tmp_path, rows)
    provider = _provider_returning([
        {"name": "anthropic-key", "type": "reference",
         "description": "Anthropic key location.",
         "body": "The worker authenticates with an anthropic key for deploys.",
         "confidence": 0.8, "likely_duplicate_of": None},
    ])
    out = distill(t, session_id="sess", provider=provider)
    assert len(out) == 1
    assert secret not in out[0].source_excerpt
    # the redaction placeholder is what survives, never the raw key
    if "anthropic" in out[0].source_excerpt.lower():
        assert "[REDACTED:anthropic]" in out[0].source_excerpt


def test_source_excerpt_appears_in_proposals_md(tmp_path):
    rows = [
        {"message": {"role": "user",
                     "content": "Production logs live in Grafana Loki, queried by service label."}},
    ]
    t = _write_transcript(tmp_path, rows)
    provider = _provider_returning([
        {"name": "logs-in-loki", "type": "reference",
         "description": "Where prod logs live.",
         "body": "Production logs are stored in Grafana Loki.",
         "confidence": 0.85, "likely_duplicate_of": None},
    ])
    out = distill(t, session_id="sess", provider=provider)
    queue = tmp_path / "queue"
    emit(out, queue)
    md = (queue / "proposals.md").read_text()
    assert "excerpt:" in md
    assert "Loki" in md


# --------------------------------------------------- time_sensitive heuristic
def test_pr_open_project_fact_is_time_sensitive():
    body = "The billing-rewrite is blocked on PR #6, which is still open."
    assert is_time_sensitive("project", body) is True


def test_branch_project_fact_is_time_sensitive():
    body = "Work continues on branch mm-w2-prov in an isolated worktree."
    assert is_time_sensitive("project", body) is True


def test_in_progress_status_is_time_sensitive():
    body = "The migration is currently running and not yet merged."
    assert is_time_sensitive("project", body) is True


def test_concrete_date_project_fact_is_time_sensitive():
    body = "The closed beta opened on 2026-05-31 with review still outstanding."
    assert is_time_sensitive("project", body) is True


def test_durable_reference_fact_is_not_time_sensitive():
    body = "Production logs live in Grafana Loki — query by the service label."
    assert is_time_sensitive("reference", body) is False


def test_durable_user_fact_is_not_time_sensitive():
    # Even a project-shaped marker on a non-project type stays durable.
    body = "The user is a staff backend engineer who owns three services."
    assert is_time_sensitive("user", body) is False


def test_plain_project_fact_without_markers_is_not_time_sensitive():
    body = "The project follows a strict no-direct-push-to-main convention."
    assert is_time_sensitive("project", body) is False


def test_time_sensitive_flag_set_through_distill_and_rendered(tmp_path):
    rows = [
        {"message": {"role": "user",
                     "content": "The deploy is blocked on PR #6 which is still open."}},
    ]
    t = _write_transcript(tmp_path, rows)
    provider = _provider_returning([
        {"name": "deploy-blocked-pr6", "type": "project",
         "description": "Deploy blocker.",
         "body": "The deploy is blocked on PR #6, which is still open.",
         "confidence": 0.7, "likely_duplicate_of": None},
    ])
    out = distill(t, session_id="sess", provider=provider)
    assert len(out) == 1
    assert out[0].time_sensitive is True

    queue = tmp_path / "queue"
    emit(out, queue)
    md = (queue / "proposals.md").read_text()
    assert "⏳" in md  # staleness marker rendered


def test_durable_fact_has_no_staleness_marker(tmp_path):
    rows = [{"message": {"role": "user",
                         "content": "Logs live in Grafana Loki."}}]
    t = _write_transcript(tmp_path, rows)
    provider = _provider_returning([
        {"name": "logs-loki", "type": "reference",
         "description": "Log location.",
         "body": "Logs live in Grafana Loki.",
         "confidence": 0.8, "likely_duplicate_of": None},
    ])
    out = distill(t, session_id="sess", provider=provider)
    assert out[0].time_sensitive is False
    queue = tmp_path / "queue"
    emit(out, queue)
    assert "⏳" not in (queue / "proposals.md").read_text()


# ---------------------------------------------------- schema + round-trip
def test_new_fields_round_trip_through_dict():
    p = MemoryProposal(
        name="x", type="project", description="d",
        body="The deploy is blocked on PR #6.",
        source_session="s", source_chunk=0,
        source_excerpt="USER: the deploy is blocked on PR #6",
        time_sensitive=True,
    )
    d = p.to_dict()
    assert d["source_excerpt"].startswith("USER:")
    assert d["time_sensitive"] is True
    p2 = MemoryProposal.from_dict(d)
    assert p2.source_excerpt == p.source_excerpt
    assert p2.time_sensitive is True
    assert p2.id == p.id


def test_new_fields_survive_jsonl_round_trip(tmp_path):
    p = MemoryProposal(
        name="x", type="project", description="d",
        body="blocked on PR #6", source_session="s", source_chunk=0,
        source_excerpt="USER: blocked on PR #6", time_sensitive=True,
    )
    queue = tmp_path / "queue"
    emit([p], queue)
    line = (queue / "proposals.jsonl").read_text().strip().splitlines()[0]
    d = json.loads(line)
    assert d["source_excerpt"] == "USER: blocked on PR #6"
    assert d["time_sensitive"] is True


def test_proposal_with_new_fields_is_schema_valid():
    ok, reason = validate_proposal({
        "name": "n", "type": "project", "description": "d",
        "body": "blocked on PR #6",
        "source_excerpt": "USER: blocked on PR #6",
        "time_sensitive": True,
        "confidence": 0.5, "likely_duplicate_of": None,
    })
    assert ok, reason


def test_bad_time_sensitive_type_rejected():
    ok, reason = validate_proposal({
        "name": "n", "type": "project", "description": "d", "body": "b",
        "time_sensitive": "yes",  # not a bool
    })
    assert not ok
    assert "time_sensitive" in reason


def test_bad_source_excerpt_type_rejected():
    ok, reason = validate_proposal({
        "name": "n", "type": "project", "description": "d", "body": "b",
        "source_excerpt": 123,  # not a string
    })
    assert not ok
    assert "source_excerpt" in reason
