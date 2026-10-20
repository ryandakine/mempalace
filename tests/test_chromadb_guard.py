"""
Tests for the ChromaDB version guard.

WHY THIS EXISTS: on 2026-10-03 the palace was opened once with a different
interpreter's chromadb 1.4.0. That single open silently rewrote the HNSW
vector segment into a 1.x on-disk format that chromadb 0.6.3 (the pinned
runtime) can no longer instantiate — semantic search died for every fresh
0.6.3 process across all wings until a full `mempalace repair`. The guard
makes every mempalace-mediated open refuse any chromadb outside the 0.6.x
line, so the installed runtime can never drift onto the store unnoticed.
"""

import ast
import sys
import types
from pathlib import Path

import chromadb
import pytest

from mempalace import config as mempalace_config
from mempalace.config import (
    UnsupportedChromaDBError,
    get_palace_collection,
    open_palace_client,
    require_supported_chromadb,
)


@pytest.mark.parametrize("version", ["0.6.3", "0.6.0", "0.6.9", "0.6.3.post1"])
def test_accepts_0_6_line(version):
    assert require_supported_chromadb(version) == version


@pytest.mark.parametrize("version", ["1.4.0", "1.0.0b0", "2.0.0", "0.5.5", "0.7.0"])
def test_rejects_other_lines(version):
    with pytest.raises(UnsupportedChromaDBError):
        require_supported_chromadb(version)


def test_error_is_a_runtime_error_and_names_both_versions():
    # Must be catchable as RuntimeError so existing `except Exception` sites
    # don't change behaviour, and must tell the operator what to do.
    with pytest.raises(RuntimeError) as exc:
        require_supported_chromadb("1.4.0")
    msg = str(exc.value)
    assert "1.4.0" in msg
    assert "0.6" in msg


def test_default_reads_installed_version():
    # The test environment runs the pinned 0.6.3, so the no-arg form passes
    # and reports the real installed version.
    assert require_supported_chromadb() == chromadb.__version__


def test_escape_hatch_env_var(monkeypatch):
    # Deliberate migrations (e.g. a planned Chroma upgrade + repair) must be
    # possible without editing the guard.
    monkeypatch.setenv("MEMPALACE_ALLOW_ANY_CHROMADB", "1")
    assert require_supported_chromadb("1.4.0") == "1.4.0"


def test_open_palace_client_refuses_before_touching_the_store(monkeypatch, tmp_path):
    # The whole point: an unsupported chromadb must never construct a client
    # (constructing one is what rewrites the on-disk segment).
    monkeypatch.setattr(chromadb, "__version__", "1.4.0")
    monkeypatch.setattr(
        chromadb,
        "PersistentClient",
        lambda **_: pytest.fail("PersistentClient constructed despite unsupported chromadb"),
    )
    with pytest.raises(UnsupportedChromaDBError):
        open_palace_client(str(tmp_path))


def test_open_palace_client_returns_a_working_client(tmp_path):
    client = open_palace_client(str(tmp_path))
    assert client.heartbeat() > 0


def test_get_palace_collection_goes_through_the_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(chromadb, "__version__", "1.4.0")
    with pytest.raises(UnsupportedChromaDBError):
        get_palace_collection(str(tmp_path), create=True)


def test_hybrid_search_keyword_mode_fails_loud(monkeypatch, tmp_path):
    # Found by Codex review: keyword mode fetched the collection inside its own
    # broad `except Exception` and returned a clean-looking `results: []`.
    from mempalace.fts_index import FTSIndex
    from mempalace.searcher import hybrid_search

    FTSIndex(str(tmp_path)).index("d1", "w", "r", "hello palace", "")
    monkeypatch.setattr(chromadb, "__version__", "1.4.0")
    with pytest.raises(UnsupportedChromaDBError):
        hybrid_search("hello", str(tmp_path), mode="keyword")


def test_hybrid_search_hybrid_mode_fails_loud(monkeypatch, tmp_path):
    from mempalace.fts_index import FTSIndex
    from mempalace.searcher import hybrid_search

    FTSIndex(str(tmp_path)).index("d1", "w", "r", "hello palace", "")
    monkeypatch.setattr(chromadb, "__version__", "1.4.0")
    with pytest.raises(UnsupportedChromaDBError):
        hybrid_search("hello", str(tmp_path), mode="hybrid")


def test_mcp_server_lookup_fails_loud_not_empty(monkeypatch):
    # The MCP server maps any lookup failure to "No palace found". The guard
    # must NOT be masked by that: an operator on the wrong chromadb needs the
    # real reason, not a mysteriously empty palace — which is exactly the
    # symptom that cost a day during the 2026-10 incident.
    from mempalace import mcp_server

    monkeypatch.setattr(chromadb, "__version__", "1.4.0")
    with pytest.raises(UnsupportedChromaDBError):
        mcp_server._get_collection()


def test_mcp_protocol_response_carries_the_guard_message(monkeypatch):
    # Found by Codex review: the generic JSON-RPC handler collapsed the raised
    # guard into "Internal tool error", so the version, interpreter and fix only
    # ever reached stderr — the MCP caller saw nothing actionable.
    from mempalace.mcp_server import handle_request

    monkeypatch.setattr(chromadb, "__version__", "1.4.0")
    resp = handle_request(
        {
            "method": "tools/call",
            "id": 7,
            "params": {"name": "mempalace_list_wings", "arguments": {}},
        }
    )
    assert "error" in resp
    msg = resp["error"]["message"]
    assert msg != "Internal tool error"
    assert "1.4.0" in msg and "0.6" in msg


# --- Coverage for every re-raise branch (found by the testing specialist) ----
# Each site below wraps the guarded open in a broad `except Exception` that
# would otherwise turn the guard into "No palace found" / an empty result.


def test_hybrid_search_fusion_lookup_error_propagates(monkeypatch, tmp_path):
    # search_memories runs first and succeeds (empty palace -> no results); the
    # fusion block's own collection lookup then fails and must surface as-is.
    # (searcher binds get_palace_collection at import time, so patch it there.)
    from mempalace import searcher
    from mempalace.fts_index import FTSIndex

    FTSIndex(str(tmp_path)).index("d1", "w", "r", "hello palace", "")
    real = searcher.get_palace_collection
    calls = {"n": 0}

    def second_call_fails(path):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("fusion lookup")
        return real(path)

    monkeypatch.setattr(searcher, "get_palace_collection", second_call_fails)
    with pytest.raises(RuntimeError, match="fusion lookup"):
        searcher.hybrid_search("hello", str(tmp_path), mode="hybrid")
    assert calls["n"] == 2


def test_searcher_search_fails_loud(monkeypatch, tmp_path):
    from mempalace.searcher import search

    monkeypatch.setattr(chromadb, "__version__", "1.4.0")
    with pytest.raises(UnsupportedChromaDBError):
        search("hello", str(tmp_path))


def test_miner_status_fails_loud(monkeypatch, tmp_path):
    from mempalace import miner

    monkeypatch.setattr(chromadb, "__version__", "1.4.0")
    with pytest.raises(UnsupportedChromaDBError):
        miner.status(str(tmp_path))


@pytest.mark.parametrize(
    "build_and_call",
    [
        lambda L, p: L.Layer1(p).generate(),
        lambda L, p: L.Layer2(p).retrieve(),
        lambda L, p: L.Layer3(p).search("hello"),
        lambda L, p: L.Layer3(p).search_raw("hello"),
        lambda L, p: L.MemoryStack(p).status(),
    ],
    ids=[
        "Layer1.generate",
        "Layer2.retrieve",
        "Layer3.search",
        "Layer3.search_raw",
        "MemoryStack.status",
    ],
)
def test_layers_fail_loud(monkeypatch, tmp_path, build_and_call):
    # These back the L1-L3 wake-up/search tools an agent calls with nobody
    # watching stderr — an empty-looking palace there has zero diagnostic signal.
    from mempalace import layers

    monkeypatch.setattr(chromadb, "__version__", "1.4.0")
    with pytest.raises(UnsupportedChromaDBError):
        build_and_call(layers, str(tmp_path))


@pytest.mark.parametrize("command", ["cmd_repair", "cmd_migrate", "cmd_compress"])
def test_cli_recovery_commands_fail_loud(monkeypatch, tmp_path, command):
    # repair/migrate/compress are what a user reaches for when the palace looks
    # broken — the exact moment the real reason must not be hidden.
    import argparse

    from mempalace import cli

    monkeypatch.setattr(chromadb, "__version__", "1.4.0")
    args = argparse.Namespace(palace=str(tmp_path), config=None)
    with pytest.raises(UnsupportedChromaDBError):
        getattr(cli, command)(args)


# --- Static contract: no call site may swallow the guard --------------------
# Palace opens go through get_palace_collection(), which propagates every error
# except a genuinely missing collection. Any site that must wrap an open in a
# broad `except` (e.g. a best-effort write) has to re-raise the guard first.

# Bare-name calls are the wrapper functions (open_palace_client, get_palace_collection,
# miner/convo_miner.get_collection). `client.get_collection(...)` is Chroma's own method,
# not an open, so attribute calls match only the two unambiguous names.
GUARDED_FUNCS = {"open_palace_client", "get_palace_collection", "get_collection"}
GUARDED_ATTRS = {"open_palace_client", "get_palace_collection"}


def _calls(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Name) and fn.id in GUARDED_FUNCS:
                yield fn.id
            elif isinstance(fn, ast.Attribute) and fn.attr in GUARDED_ATTRS:
                yield fn.attr


def _is_broad(handler):
    t = handler.type
    if t is None:
        return True
    names = [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
    return "Exception" in names or "BaseException" in names


def _is_specific(handler):
    t = handler.type
    return t is not None and any(
        isinstance(n, ast.Name) and n.id == "UnsupportedChromaDBError" for n in ast.walk(t)
    )


def test_every_guarded_open_reraises_before_any_broad_except():
    pkg = Path(mempalace_config.__file__).parent
    offenders = []
    for py in sorted(pkg.rglob("*.py")):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            if not any(_calls(ast.Module(body=node.body, type_ignores=[]))):
                continue
            seen_specific = False
            for h in node.handlers:
                if _is_specific(h):
                    seen_specific = True
                elif _is_broad(h) and not seen_specific:
                    offenders.append(f"{py.name}:{h.lineno}")
                    break
    assert not offenders, (
        "guarded palace open inside a try whose broad `except` would swallow "
        f"UnsupportedChromaDBError — add `except UnsupportedChromaDBError: raise` first: {offenders}"
    )


def test_memory_miner_dual_write_fails_loud(monkeypatch):
    # Found by the adversarial pass: the write side swallowed the guard too, in a
    # subpackage the contract test originally did not scan.
    from mempalace.memory_miner.accept import _dual_write

    monkeypatch.setattr(chromadb, "__version__", "1.4.0")
    with pytest.raises(UnsupportedChromaDBError):
        _dual_write({"id": "x", "body": "hello"}, "notes.md")


# --- Found by the red team: a chromadb that cannot be imported at all ------
# Every re-raise above is specific to UnsupportedChromaDBError; a bare
# ImportError/AttributeError from the guard's own `import chromadb` would have
# fallen through the broad handlers into "No palace found".


def test_guard_surfaces_chromadb_import_failure(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "chromadb", None)  # makes `import chromadb` raise
    with pytest.raises(UnsupportedChromaDBError) as exc:
        require_supported_chromadb()
    assert "could not be imported" in str(exc.value)
    with pytest.raises(UnsupportedChromaDBError):
        open_palace_client(str(tmp_path))


def test_guard_surfaces_missing_version_attribute(monkeypatch):
    stub = types.ModuleType("chromadb")  # imports fine, but no __version__
    monkeypatch.setitem(sys.modules, "chromadb", stub)
    with pytest.raises(UnsupportedChromaDBError):
        require_supported_chromadb()


def test_cli_repair_surfaces_import_failure(monkeypatch, tmp_path):
    import argparse

    from mempalace import cli

    monkeypatch.setitem(sys.modules, "chromadb", None)
    with pytest.raises(UnsupportedChromaDBError):
        cli.cmd_repair(argparse.Namespace(palace=str(tmp_path), config=None))


def test_mcp_boundary_surfaces_import_failure_not_no_palace(monkeypatch):
    # Before the fix this returned a *result* of {"error": "No palace found",
    # "hint": "Run: mempalace init ... && mempalace mine ..."} — advice that would
    # send an operator off to re-mine instead of fixing their Python environment.
    from mempalace.mcp_server import handle_request

    monkeypatch.setitem(sys.modules, "chromadb", None)
    resp = handle_request(
        {
            "method": "tools/call",
            "id": 9,
            "params": {"name": "mempalace_list_wings", "arguments": {}},
        }
    )
    assert "result" not in resp
    assert "could not be imported" in resp["error"]["message"]


def test_guard_surfaces_import_time_runtime_error(monkeypatch):
    # chromadb raises a plain RuntimeError at import when the system sqlite is too
    # old; that must surface as the guard's error too, not vanish into "no palace".
    class Exploding(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError("Your system has an unsupported version of sqlite3")

    monkeypatch.setitem(sys.modules, "chromadb", Exploding("chromadb"))
    with pytest.raises(UnsupportedChromaDBError) as exc:
        require_supported_chromadb()
    assert "sqlite3" in str(exc.value)


def test_accept_preflights_guard_before_any_write(monkeypatch, tmp_path):
    # Found by the final Codex pass: re-raising from _dual_write fired AFTER the
    # memory file, index and ledger were committed, so a retry after fixing
    # chromadb skipped the proposal as already accepted and its palace copy was
    # lost. With dual-write on, the guard now runs before any mutation.
    from mempalace.memory_miner.accept import accept

    queue = tmp_path / "queue.jsonl"
    queue.write_text("")
    memory_dir = tmp_path / "memory"
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(chromadb, "__version__", "1.4.0")
    with pytest.raises(UnsupportedChromaDBError):
        accept(
            str(queue),
            memory_dir=memory_dir,
            ledger_path=ledger,
            index_path=memory_dir / "MEMORY.md",
            dual_write_mempalace=True,
        )
    assert not ledger.exists()
    assert not memory_dir.exists() or not list(memory_dir.glob("*.md"))


# --- Every chromadb READ error fails loud (not just the version guard) -------
# get_palace_collection() returns None only when the collection does not exist;
# a corrupt or unreadable store must surface as its real error at every site,
# never as "No palace found".


class _CorruptClient:
    def get_collection(self, name):
        raise RuntimeError("corrupt segment")


class _EmptyClient:
    def get_collection(self, name):
        from chromadb.errors import InvalidCollectionException

        raise InvalidCollectionException(f"Collection {name} does not exist.")


def test_get_palace_collection_none_only_for_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(mempalace_config, "open_palace_client", lambda p: _EmptyClient())
    assert get_palace_collection(str(tmp_path)) is None
    monkeypatch.setattr(mempalace_config, "open_palace_client", lambda p: _CorruptClient())
    with pytest.raises(RuntimeError, match="corrupt segment"):
        get_palace_collection(str(tmp_path))


def test_fresh_palace_dir_reads_as_missing_not_error(tmp_path):
    assert get_palace_collection(str(tmp_path)) is None


@pytest.mark.parametrize(
    "call",
    [
        lambda m, p: m.layers.Layer1(p).generate(),
        lambda m, p: m.layers.Layer2(p).retrieve(),
        lambda m, p: m.layers.Layer3(p).search("hello"),
        lambda m, p: m.layers.Layer3(p).search_raw("hello"),
        lambda m, p: m.layers.MemoryStack(p).status(),
        lambda m, p: m.miner.status(p),
        lambda m, p: m.searcher.search("hello", p),
        lambda m, p: m.searcher.search_memories("hello", p),
        lambda m, p: m.cli.cmd_repair(m.argparse.Namespace(palace=p, config=None)),
        lambda m, p: m.cli.cmd_migrate(m.argparse.Namespace(palace=p, config=None)),
        lambda m, p: m.cli.cmd_compress(m.argparse.Namespace(palace=p, config=None)),
        lambda m, p: m.mcp_server._get_collection(),
    ],
    ids=[
        "Layer1.generate",
        "Layer2.retrieve",
        "Layer3.search",
        "Layer3.search_raw",
        "MemoryStack.status",
        "miner.status",
        "searcher.search",
        "searcher.search_memories",
        "cli.repair",
        "cli.migrate",
        "cli.compress",
        "mcp._get_collection",
    ],
)
def test_corrupt_store_fails_loud_at_every_read_site(monkeypatch, tmp_path, call):
    import argparse
    import types as _types

    from mempalace import cli, layers, mcp_server, miner, searcher

    monkeypatch.setattr(mempalace_config, "open_palace_client", lambda p: _CorruptClient())
    mods = _types.SimpleNamespace(
        cli=cli,
        layers=layers,
        mcp_server=mcp_server,
        miner=miner,
        searcher=searcher,
        argparse=argparse,
    )
    with pytest.raises(RuntimeError, match="corrupt segment"):
        call(mods, str(tmp_path))


def test_mcp_boundary_carries_real_error_for_any_read_failure(monkeypatch):
    # Not just the version guard: a corrupt store must reach the MCP caller as
    # its real type and message, not an opaque "Internal tool error".
    from mempalace.mcp_server import handle_request

    monkeypatch.setattr(mempalace_config, "open_palace_client", lambda p: _CorruptClient())
    resp = handle_request(
        {
            "method": "tools/call",
            "id": 11,
            "params": {"name": "mempalace_list_wings", "arguments": {}},
        }
    )
    assert "result" not in resp
    assert resp["error"]["message"] == "RuntimeError: corrupt segment"


class _PostOpenFailingCollection:
    def count(self):
        raise RuntimeError("post-open failure")

    def get(self, *a, **k):
        raise RuntimeError("post-open failure")

    def query(self, *a, **k):
        raise RuntimeError("post-open failure")


class _OpensButFailsLaterClient:
    def get_collection(self, name):
        return _PostOpenFailingCollection()


@pytest.mark.parametrize(
    "call",
    [
        lambda m, p: m.cli.cmd_repair(m.argparse.Namespace(palace=p, config=None)),
        lambda m, p: m.layers.MemoryStack(p).status(),
    ],
    ids=["cli.repair:count", "MemoryStack.status:count"],
)
def test_post_open_failures_propagate(monkeypatch, tmp_path, call):
    # The open succeeds; the first real read on the collection fails. Nothing
    # downstream of the open may swallow that either.
    import argparse
    import types as _types

    from mempalace import cli, layers, searcher

    monkeypatch.setattr(
        mempalace_config, "open_palace_client", lambda p: _OpensButFailsLaterClient()
    )
    mods = _types.SimpleNamespace(cli=cli, layers=layers, searcher=searcher, argparse=argparse)
    with pytest.raises(RuntimeError, match="post-open failure"):
        call(mods, str(tmp_path))


def test_cli_entry_point_prints_real_error_one_line(monkeypatch, tmp_path, capsys):
    # Found by the adversarial re-check: with the broad excepts gone, the CLI
    # dumped a 30-line chromadb traceback on a corrupt store — including from
    # `mempalace repair`. Loud means the real reason on one line, exit 1.
    from mempalace import cli

    monkeypatch.setattr(mempalace_config, "open_palace_client", lambda p: _CorruptClient())
    monkeypatch.setattr(sys, "argv", ["mempalace", "--palace", str(tmp_path), "status"])
    monkeypatch.delenv("MEMPALACE_DEBUG", raising=False)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Error: RuntimeError: corrupt segment" in err
    assert "Traceback" not in err


def test_cli_entry_point_reraises_in_debug(monkeypatch, tmp_path):
    from mempalace import cli

    monkeypatch.setattr(mempalace_config, "open_palace_client", lambda p: _CorruptClient())
    monkeypatch.setattr(sys, "argv", ["mempalace", "--palace", str(tmp_path), "status"])
    monkeypatch.setenv("MEMPALACE_DEBUG", "1")
    with pytest.raises(RuntimeError, match="corrupt segment"):
        cli.main()


def test_search_memories_returns_real_error_on_query_failure(monkeypatch, tmp_path):
    # search_memories has a structured error channel (the MCP search tool hands
    # it back as the tool result); a query failure must carry the real message
    # there — loud, just not as an exception.
    from mempalace import searcher

    monkeypatch.setattr(
        mempalace_config, "open_palace_client", lambda p: _OpensButFailsLaterClient()
    )
    resp = searcher.search_memories("hello", str(tmp_path))
    assert resp["error"] == "Search error: post-open failure"
