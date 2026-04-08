#!/usr/bin/env python3
"""
searcher.py — Find anything. Exact words.

Semantic search against the palace.
Returns verbatim text — the actual words, never summaries.
"""

import logging
import os
from pathlib import Path

import chromadb

logger = logging.getLogger("mempalace_mcp")


class SearchError(Exception):
    """Raised when search cannot proceed (e.g. no palace found)."""


def search(query: str, palace_path: str, wing: str = None, room: str = None, n_results: int = 5):
    """
    Search the palace. Returns verbatim drawer content.
    Optionally filter by wing (project) or room (aspect).
    """
    try:
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
    except Exception:
        print(f"\n  No palace found at {palace_path}")
        print("  Run: mempalace init <dir> then mempalace mine <dir>")
        raise SearchError(f"No palace found at {palace_path}")

    # Build where filter
    where = {}
    if wing and room:
        where = {"$and": [{"wing": wing}, {"room": room}]}
    elif wing:
        where = {"wing": wing}
    elif room:
        where = {"room": room}

    try:
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

    except Exception as e:
        print(f"\n  Search error: {e}")
        raise SearchError(f"Search error: {e}") from e

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not docs:
        print(f'\n  No results found for: "{query}"')
        return

    print(f"\n{'=' * 60}")
    print(f'  Results for: "{query}"')
    if wing:
        print(f"  Wing: {wing}")
    if room:
        print(f"  Room: {room}")
    print(f"{'=' * 60}\n")

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        similarity = round(1 - dist, 3)
        source = Path(meta.get("source_file", "?")).name
        wing_name = meta.get("wing", "?")
        room_name = meta.get("room", "?")

        print(f"  [{i}] {wing_name} / {room_name}")
        print(f"      Source: {source}")
        print(f"      Match:  {similarity}")
        print()
        # Print the verbatim text, indented
        for line in doc.strip().split("\n"):
            print(f"      {line}")
        print()
        print(f"  {'─' * 56}")

    print()


def search_memories(
    query: str, palace_path: str, wing: str = None, room: str = None, n_results: int = 5
) -> dict:
    """
    Programmatic search — returns a dict instead of printing.
    Used by the MCP server and other callers that need data.
    """
    try:
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
    except Exception as e:
        logger.error("No palace found at %s: %s", palace_path, e)
        return {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }

    # Build where filter
    where = {}
    if wing and room:
        where = {"$and": [{"wing": wing}, {"room": room}]}
    elif wing:
        where = {"wing": wing}
    elif room:
        where = {"room": room}

    try:
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)
    except Exception as e:
        return {"error": f"Search error: {e}"}

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "text": doc,
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(meta.get("source_file", "?")).name,
                "similarity": round(1 - dist, 3),
            }
        )

    return {
        "query": query,
        "filters": {"wing": wing, "room": room},
        "results": hits,
    }


def reciprocal_rank_fusion(
    semantic_results: list,
    keyword_results: list,
    k: int = 60,
    semantic_weight: float = 0.5,
    keyword_weight: float = 0.5,
) -> list:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    RRF score = sum over each list: weight / (k + rank).
    Items appearing in both lists score higher than items in only one.
    """
    scores = {}

    for rank_0, result in enumerate(semantic_results):
        did = result.get("drawer_id") or result.get("id", str(rank_0))
        scores[did] = scores.get(did, 0.0) + semantic_weight / (k + rank_0 + 1)

    for rank_0, result in enumerate(keyword_results):
        did = result.get("drawer_id", str(rank_0))
        scores[did] = scores.get(did, 0.0) + keyword_weight / (k + rank_0 + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
    mode: str = "hybrid",
) -> dict:
    """Search using semantic, keyword, or hybrid (fused) mode.

    Args:
        mode: "hybrid" (default), "semantic", or "keyword"
    """
    from .fts_index import FTSIndex

    fts = None
    fts_path = os.path.join(palace_path, "mempalace_fts.sqlite3")
    if os.path.exists(fts_path):
        fts = FTSIndex(palace_path)

    # Keyword-only mode
    if mode == "keyword":
        if not fts or not fts.exists():
            return {
                "query": query,
                "mode": "keyword",
                "error": "No FTS index found. Run: mempalace rebuild-index",
                "results": [],
            }
        kw_results = fts.search(query, wing=wing, room=room, limit=n_results)
        # Fetch full documents from ChromaDB for keyword hits
        try:
            from .config import get_palace_collection
            col = get_palace_collection(palace_path)
            if col and kw_results:
                ids = [r["drawer_id"] for r in kw_results]
                full = col.get(ids=ids, include=["documents", "metadatas"])
                doc_map = {did: (doc, meta) for did, doc, meta in zip(
                    full["ids"], full["documents"], full["metadatas"]
                )}
                hits = []
                for r in kw_results:
                    doc, meta = doc_map.get(r["drawer_id"], ("", {}))
                    hits.append({
                        "text": doc,
                        "wing": meta.get("wing", "unknown"),
                        "room": meta.get("room", "unknown"),
                        "source_file": Path(meta.get("source_file", "?")).name,
                        "match_type": "keyword",
                        "keyword_snippet": r.get("snippet", ""),
                    })
                return {"query": query, "mode": "keyword", "results": hits}
        except Exception:
            pass
        return {"query": query, "mode": "keyword", "results": []}

    # Semantic-only mode (or hybrid fallback when no FTS)
    semantic_result = search_memories(query, palace_path, wing=wing, room=room, n_results=n_results if mode == "semantic" else n_results * 2)

    if mode == "semantic" or not fts or not fts.exists():
        if mode == "hybrid" and (not fts or not fts.exists()):
            logger.info("No FTS index found. Falling back to semantic-only. Run: mempalace rebuild-index")
        semantic_result["mode"] = mode if fts else "semantic"
        return semantic_result

    # Hybrid mode: fuse semantic + keyword
    fetch_limit = n_results * 2
    kw_results = fts.search(query, wing=wing, room=room, limit=fetch_limit)

    # Build ID-indexed lookup for semantic results
    semantic_hits = semantic_result.get("results", [])
    # Generate drawer_id approximations from semantic results for fusion
    for i, hit in enumerate(semantic_hits):
        hit["drawer_id"] = f"sem_{i}"  # placeholder, will use position-based fusion

    # Build keyword ID set for match_type detection
    kw_ids = {r["drawer_id"] for r in kw_results}
    kw_snippets = {r["drawer_id"]: r.get("snippet", "") for r in kw_results}

    # RRF fusion
    fused = reciprocal_rank_fusion(semantic_hits, kw_results)

    # Collect top N results with full text
    # Semantic results already have full text; keyword results need ChromaDB lookup
    try:
        from .config import get_palace_collection
        col = get_palace_collection(palace_path)
        kw_doc_map = {}
        if col and kw_results:
            kw_ids_list = [r["drawer_id"] for r in kw_results]
            full = col.get(ids=kw_ids_list, include=["documents", "metadatas"])
            kw_doc_map = {did: (doc, meta) for did, doc, meta in zip(
                full["ids"], full["documents"], full["metadatas"]
            )}
    except Exception:
        kw_doc_map = {}

    hits = []
    sem_by_id = {h["drawer_id"]: h for h in semantic_hits}

    for drawer_id, fused_score in fused[:n_results]:
        if drawer_id in sem_by_id:
            h = sem_by_id[drawer_id]
            match_type = "both" if drawer_id in kw_ids else "semantic"
            hits.append({
                "text": h["text"],
                "wing": h["wing"],
                "room": h["room"],
                "source_file": h.get("source_file", "?"),
                "similarity": h.get("similarity", 0),
                "relevance": round(fused_score, 4),
                "match_type": match_type,
            })
        elif drawer_id in kw_doc_map:
            doc, meta = kw_doc_map[drawer_id]
            hits.append({
                "text": doc,
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(meta.get("source_file", "?")).name,
                "relevance": round(fused_score, 4),
                "match_type": "keyword",
                "keyword_snippet": kw_snippets.get(drawer_id, ""),
            })

    return {
        "query": query,
        "mode": "hybrid",
        "results": hits,
    }
