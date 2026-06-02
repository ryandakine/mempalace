"""
fts_index.py — SQLite FTS5 full-text search index for BM25 keyword retrieval.

Runs parallel to ChromaDB. Same drawer IDs, same content.
Provides keyword search that complements semantic search.
"""

import os
import re
import sqlite3
import logging

logger = logging.getLogger("mempalace_mcp")


class FTSIndex:
    """SQLite FTS5 full-text search index for keyword/BM25 retrieval."""

    def __init__(self, palace_path: str):
        self.db_path = os.path.join(palace_path, "mempalace_fts.sqlite3")
        self._init_db()

    def _init_db(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS drawer_text (
                drawer_id   TEXT PRIMARY KEY,
                wing        TEXT NOT NULL,
                room        TEXT NOT NULL,
                content     TEXT NOT NULL,
                filed_at    TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_drawer_wing ON drawer_text(wing);
            CREATE INDEX IF NOT EXISTS idx_drawer_room ON drawer_text(room);
        """)
        # FTS5 virtual table with external content
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS drawer_fts USING fts5(
                    content,
                    content='drawer_text',
                    content_rowid='rowid',
                    tokenize='unicode61 remove_diacritics 2'
                )
            """)
        except sqlite3.OperationalError:
            # FTS5 might already exist
            pass

        # Auto-sync triggers
        for trigger_sql in [
            """CREATE TRIGGER IF NOT EXISTS drawer_text_ai AFTER INSERT ON drawer_text BEGIN
                INSERT INTO drawer_fts(rowid, content) VALUES (new.rowid, new.content);
            END""",
            """CREATE TRIGGER IF NOT EXISTS drawer_text_ad AFTER DELETE ON drawer_text BEGIN
                INSERT INTO drawer_fts(drawer_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END""",
            """CREATE TRIGGER IF NOT EXISTS drawer_text_au AFTER UPDATE ON drawer_text BEGIN
                INSERT INTO drawer_fts(drawer_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
                INSERT INTO drawer_fts(rowid, content) VALUES (new.rowid, new.content);
            END""",
        ]:
            try:
                conn.execute(trigger_sql)
            except sqlite3.OperationalError:
                pass

        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def index(self, drawer_id: str, wing: str, room: str, content: str, filed_at: str = ""):
        """Index a drawer's content for full-text search."""
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO drawer_text (drawer_id, wing, room, content, filed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (drawer_id, wing, room, content, filed_at),
        )
        conn.commit()
        conn.close()

    def index_batch(self, items: list):
        """Batch index multiple drawers. Each item is a dict with drawer_id, wing, room, content, filed_at."""
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO drawer_text (drawer_id, wing, room, content, filed_at) "
            "VALUES (:drawer_id, :wing, :room, :content, :filed_at)",
            items,
        )
        conn.commit()
        conn.close()

    def delete(self, drawer_id: str):
        """Remove a drawer from the FTS index."""
        conn = self._conn()
        conn.execute("DELETE FROM drawer_text WHERE drawer_id = ?", (drawer_id,))
        conn.commit()
        conn.close()

    def search(self, query: str, wing: str = None, room: str = None, limit: int = 20) -> list:
        """BM25 keyword search. Returns list of dicts with drawer_id, bm25_rank, snippet."""
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []

        sql = """
            SELECT dt.drawer_id, bm25(drawer_fts) as rank,
                   snippet(drawer_fts, 0, '>>>', '<<<', '...', 40) as snippet
            FROM drawer_fts
            JOIN drawer_text dt ON drawer_fts.rowid = dt.rowid
        """
        params = [fts_query]
        wheres = ["drawer_fts MATCH ?"]

        if wing:
            wheres.append("dt.wing = ?")
            params.append(wing)
        if room:
            wheres.append("dt.room = ?")
            params.append(room)

        sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        conn = self._conn()
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # Query syntax error (special chars, empty tokens, etc.)
            conn.close()
            return []
        conn.close()

        return [{"drawer_id": r[0], "bm25_rank": r[1], "snippet": r[2]} for r in rows]

    def count(self) -> int:
        """Total indexed documents."""
        conn = self._conn()
        count = conn.execute("SELECT COUNT(*) FROM drawer_text").fetchone()[0]
        conn.close()
        return count

    def rebuild(self, chromadb_collection):
        """Full rebuild of FTS index from ChromaDB collection."""
        total = chromadb_collection.count()
        if total == 0:
            return 0

        conn = self._conn()
        conn.execute("DELETE FROM drawer_text")
        conn.commit()

        offset = 0
        batch_size = 1000
        indexed = 0
        while offset < total:
            batch = chromadb_collection.get(
                limit=batch_size, offset=offset, include=["documents", "metadatas"]
            )
            items = []
            for doc_id, doc, meta in zip(batch["ids"], batch["documents"], batch["metadatas"]):
                items.append({
                    "drawer_id": doc_id,
                    "wing": meta.get("wing", "unknown"),
                    "room": meta.get("room", "unknown"),
                    "content": doc,
                    "filed_at": meta.get("filed_at", ""),
                })
            if items:
                conn.executemany(
                    "INSERT OR REPLACE INTO drawer_text (drawer_id, wing, room, content, filed_at) "
                    "VALUES (:drawer_id, :wing, :room, :content, :filed_at)",
                    items,
                )
                conn.commit()
                indexed += len(items)

            if len(batch["ids"]) < batch_size:
                break
            offset += batch_size

        return indexed

    def check_consistency(self, chromadb_collection) -> dict:
        """Compare ChromaDB drawer count vs FTS row count."""
        chroma_count = chromadb_collection.count()
        fts_count = self.count()
        return {
            "chromadb_count": chroma_count,
            "fts_count": fts_count,
            "consistent": chroma_count == fts_count,
            "drift": abs(chroma_count - fts_count),
        }

    def exists(self) -> bool:
        """Check if the FTS database file exists and has data."""
        if not os.path.exists(self.db_path):
            return False
        return self.count() > 0

    def _build_fts_query(self, user_query: str) -> str:
        """Convert user query to FTS5 query syntax.

        Handles: multi-word queries (implicit AND), quoted phrases, special chars.
        """
        # Preserve quoted phrases
        phrases = re.findall(r'"[^"]*"', user_query)
        remaining = re.sub(r'"[^"]*"', '', user_query)

        # Tokenize remaining, strip FTS5 special chars
        tokens = []
        for word in remaining.split():
            clean = re.sub(r'[^\w]', '', word)
            if clean:
                tokens.append(clean)

        all_parts = tokens + phrases
        if not all_parts:
            return ""

        return " ".join(all_parts)
