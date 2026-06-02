"""Tests for FTS5 full-text search index."""

import os
import tempfile
import shutil

import pytest

from mempalace.fts_index import FTSIndex


@pytest.fixture
def fts_dir():
    d = tempfile.mkdtemp(prefix="mempalace_fts_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def fts(fts_dir):
    return FTSIndex(fts_dir)


class TestFTSIndex:
    def test_creates_db(self, fts, fts_dir):
        assert os.path.exists(os.path.join(fts_dir, "mempalace_fts.sqlite3"))

    def test_index_and_count(self, fts):
        fts.index("d1", "wing_a", "room_x", "hello world", "2026-01-01")
        assert fts.count() == 1

    def test_index_upsert(self, fts):
        fts.index("d1", "wing_a", "room_x", "hello world", "2026-01-01")
        fts.index("d1", "wing_a", "room_x", "updated content", "2026-01-02")
        assert fts.count() == 1

    def test_delete(self, fts):
        fts.index("d1", "wing_a", "room_x", "hello world", "")
        fts.delete("d1")
        assert fts.count() == 0

    def test_batch_index(self, fts):
        items = [
            {"drawer_id": f"d{i}", "wing": "w", "room": "r", "content": f"text {i}", "filed_at": ""}
            for i in range(10)
        ]
        fts.index_batch(items)
        assert fts.count() == 10

    def test_exists_empty(self, fts):
        assert not fts.exists()

    def test_exists_with_data(self, fts):
        fts.index("d1", "w", "r", "content", "")
        assert fts.exists()


class TestFTSSearch:
    def test_basic_search(self, fts):
        fts.index("d1", "wing_a", "room_x", "The quick brown fox jumps over the lazy dog", "")
        fts.index("d2", "wing_a", "room_y", "Python programming language guide", "")
        results = fts.search("fox")
        assert len(results) == 1
        assert results[0]["drawer_id"] == "d1"

    def test_multi_word_search(self, fts):
        fts.index("d1", "w", "r", "docker compose nvidia GPU transcoding setup", "")
        fts.index("d2", "w", "r", "python flask web application deployment", "")
        results = fts.search("docker nvidia")
        assert len(results) == 1
        assert results[0]["drawer_id"] == "d1"

    def test_wing_filter(self, fts):
        fts.index("d1", "media", "infra", "docker compose setup", "")
        fts.index("d2", "poker", "hands", "docker poker bot", "")
        results = fts.search("docker", wing="media")
        assert len(results) == 1
        assert results[0]["drawer_id"] == "d1"

    def test_room_filter(self, fts):
        fts.index("d1", "w", "backend", "database migration", "")
        fts.index("d2", "w", "frontend", "database visualization", "")
        results = fts.search("database", room="backend")
        assert len(results) == 1
        assert results[0]["drawer_id"] == "d1"

    def test_bm25_ranking(self, fts):
        fts.index("d1", "w", "r", "poker poker poker strategy", "")
        fts.index("d2", "w", "r", "poker is a card game", "")
        results = fts.search("poker")
        assert len(results) == 2
        # d1 has higher term frequency, should rank first (more negative bm25)
        assert results[0]["drawer_id"] == "d1"

    def test_empty_results(self, fts):
        fts.index("d1", "w", "r", "hello world", "")
        results = fts.search("nonexistent")
        assert len(results) == 0

    def test_special_chars_dont_crash(self, fts):
        fts.index("d1", "w", "r", "test $500 bankroll 3-bet", "")
        results = fts.search("$500")
        # Should not crash, may or may not find results depending on tokenization
        assert isinstance(results, list)

    def test_empty_query(self, fts):
        results = fts.search("")
        assert results == []

    def test_limit(self, fts):
        for i in range(20):
            fts.index(f"d{i}", "w", "r", f"common word document {i}", "")
        results = fts.search("common", limit=5)
        assert len(results) == 5

    def test_snippet_contains_markers(self, fts):
        fts.index("d1", "w", "r", "The JohnDoe123 player raised preflop", "")
        results = fts.search("JohnDoe123")
        assert len(results) == 1
        assert ">>>" in results[0]["snippet"]
        assert "<<<" in results[0]["snippet"]


class TestRRF:
    def test_rrf_basic(self):
        from mempalace.searcher import reciprocal_rank_fusion

        semantic = [{"drawer_id": "a"}, {"drawer_id": "b"}]
        keyword = [{"drawer_id": "b"}, {"drawer_id": "c"}]
        fused = reciprocal_rank_fusion(semantic, keyword)

        # "b" appears in both lists, should rank highest
        ids = [did for did, score in fused]
        assert ids[0] == "b"

    def test_rrf_weights(self):
        from mempalace.searcher import reciprocal_rank_fusion

        semantic = [{"drawer_id": "a"}]
        keyword = [{"drawer_id": "b"}]
        fused = reciprocal_rank_fusion(
            semantic, keyword, semantic_weight=0.9, keyword_weight=0.1
        )
        ids = [did for did, score in fused]
        # "a" has 0.9 weight, "b" has 0.1 weight, both at rank 1
        assert ids[0] == "a"

    def test_rrf_empty_lists(self):
        from mempalace.searcher import reciprocal_rank_fusion

        fused = reciprocal_rank_fusion([], [])
        assert fused == []
