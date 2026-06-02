"""
cli.py — `python -m mempalace.memory_miner` entry point.

Subcommands:
  run     scan new transcripts → distill → dedup → emit proposals (NO live writes)
  accept  promote queued proposals into the live store (the ONLY writer)
  status  show watermark + queue summary

Default provider = grok. Default sink for accept = the auto-memory .md store.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

from . import accept as accept_mod
from . import emit as emit_mod
from .dedup import StoreIndex, classify
from .distill import distill
from .providers import get_provider
from .watermark import (
    AlreadyRunning,
    Watermark,
    count_records,
    discover_transcripts,
    parse_since,
    run_lock,
)

log = logging.getLogger("memory_miner")


def default_transcript_root() -> Path:
    return Path(os.path.expanduser("~")) / ".claude" / "projects"


def default_out_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".claude" / "memory-miner"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


# --------------------------------------------------------------------- run
def cmd_run(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = Path(args.memory_dir)
    queue_path = out_dir / "proposals.jsonl"
    ledger_path = Path(args.ledger) if args.ledger else (memory_dir.parent / "memory-miner-accepted.jsonl")

    provider_fn = get_provider(args.provider)
    if args.provider == "none":
        log.info("provider=none is diagnostic-only; it never distills or writes.")

    since_ts = None
    if args.since:
        try:
            since_ts = parse_since(args.since)
        except ValueError as e:
            log.error("%s", e)
            return 2

    index = StoreIndex.from_dir(memory_dir)
    existing_index = index.index_lines()

    roots = [Path(r) for r in (args.root or [str(default_transcript_root())])]
    transcripts = discover_transcripts(
        roots, since_ts=since_ts, newest_first=args.newest_first
    )

    try:
        lock_ctx = run_lock()
        lock_ctx.__enter__()
    except AlreadyRunning as e:
        log.error("not starting: %s", e)
        return 3

    wm = Watermark()
    total_new = 0
    processed = 0
    try:
        for t in transcripts:
            should, _skip = wm.needs_processing(t)
            if not should:
                continue
            if args.limit and processed >= args.limit:
                break
            processed += 1

            proposals = distill(
                t,
                session_id=t.stem,
                provider=provider_fn,
                existing_index=existing_index,
                today=str(date.today()),
                timeout=args.timeout,
            )
            for p in proposals:
                classify(p, index)

            written = emit_mod.emit(proposals, out_dir, accepted_ledger=ledger_path)
            total_new += len(written)
            # per-file watermark commit (plan §4.3): commit AFTER emit succeeds.
            wm.commit(t, count_records(t))
            if proposals:
                log.info("%s → %d proposals (%d new)", t.name, len(proposals), len(written))
    finally:
        lock_ctx.__exit__(None, None, None)

    print(f"\nProcessed {processed} new/changed transcript(s).")
    print(f"New proposals queued: {total_new}")
    print(f"Queue: {queue_path}")
    print(f"Review: {out_dir / 'proposals.md'}")
    print("NO memory files were written. Use `accept` to promote proposals.")
    return 0


# ------------------------------------------------------------------ accept
def cmd_accept(args) -> int:
    out_dir = Path(args.out_dir)
    queue_path = out_dir / "proposals.jsonl"
    memory_dir = Path(args.memory_dir)
    ledger_path = Path(args.ledger) if args.ledger else (memory_dir.parent / "memory-miner-accepted.jsonl")

    if not args.ids and args.above is None:
        log.error("accept needs <id>... or --above <conf>")
        return 2

    results = accept_mod.accept(
        queue_path,
        ids=args.ids or None,
        above=args.above,
        type_filter=args.type,
        memory_dir=memory_dir,
        ledger_path=ledger_path,
        dual_write_mempalace=args.dual_write_mempalace,
    )
    if not results:
        print("Nothing accepted (no matching proposals, or all already accepted).")
        return 0
    for r in results:
        print(f"  accepted {r['id']}  [{r['type']}]  → {r['file']}")
    print(f"\n{len(results)} memory file(s) written to {memory_dir}")
    return 0


# ------------------------------------------------------------------ status
def cmd_status(args) -> int:
    out_dir = Path(args.out_dir)
    queue_path = out_dir / "proposals.jsonl"
    memory_dir = Path(args.memory_dir)
    ledger_path = Path(args.ledger) if args.ledger else (memory_dir.parent / "memory-miner-accepted.jsonl")

    wm = Watermark()
    queue_ids = emit_mod.load_existing_ids(queue_path)
    accepted_ids = emit_mod.load_accepted_ids(ledger_path)
    print(f"Watermarked transcripts: {len(wm.data)}")
    print(f"Queued proposals:        {len(queue_ids)}")
    print(f"Accepted (ledger):       {len(accepted_ids)}")
    print(f"Pending review:          {len(queue_ids - accepted_ids)}")
    print(f"Queue file:              {queue_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m mempalace.memory_miner")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--out-dir", default=str(default_out_dir()),
                   help="proposal queue dir (default ~/.claude/memory-miner)")
    p.add_argument("--memory-dir", default=str(accept_mod.default_memory_dir()),
                   help="live auto-memory store dir")
    p.add_argument("--ledger", default="", help="accepted-ledger path (default: alongside memory-dir)")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="scan→distill→dedup→emit (no live writes)")
    r.add_argument("--provider", default="grok", choices=["grok", "local", "claude", "none"])
    r.add_argument("--root", action="append", help="transcript root(s) (default ~/.claude/projects)")
    r.add_argument("--limit", type=int, default=0, help="max new transcripts this run (0=all)")
    r.add_argument("--since", default="",
                   help="only mine transcripts with mtime >= YYYY-MM-DD (backlog scoping)")
    r.add_argument("--newest-first", action="store_true",
                   help="with --limit, mine the N most RECENT transcripts (mtime desc)")
    r.add_argument("--timeout", type=float, default=180.0)
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("accept", help="promote proposals into the live store")
    a.add_argument("ids", nargs="*", help="proposal id(s) to accept")
    a.add_argument("--above", type=float, help="accept all proposals with confidence >= N")
    a.add_argument("--type", choices=["user", "feedback", "project", "reference"],
                   help="restrict --above to one store type")
    a.add_argument("--dual-write-mempalace", action="store_true",
                   help="also write to MemPalace (OFF by default)")
    a.set_defaults(func=cmd_accept)

    s = sub.add_parser("status", help="show watermark + queue summary")
    s.set_defaults(func=cmd_status)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
