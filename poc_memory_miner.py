#!/usr/bin/env python3
"""
poc_memory_miner.py — Phase 0 proof-of-concept for the self-evolving memory miner.

ZERO WRITES. Points the pipeline at ONE Claude Code transcript (default: the current
session) and prints exactly what durable memories it WOULD propose, plus the metrics
the plan's §5 gate needs (prefilter cut ratio + regex->store type overlap).

Pipeline (plan §4, steps 1-5):
  1 load     this session's .jsonl
  2 normalize  reuse mempalace/normalize.py  (Claude Code jsonl -> transcript)
  3 prefilter  reuse mempalace/general_extractor.py  (FREE regex, 5 types)
  4 scrub      deterministic secret redaction BEFORE any provider sees text
  5 distill    Grok via local proxy -> validated MemoryProposal[]  (off CC subscription)

Reuses existing code by file-path import to avoid mempalace/__init__ (chromadb).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
MEM_DIR = Path.home() / ".claude" / "projects" / "-home-ryan" / "memory"
GROK_PROXY = "http://127.0.0.1:8765/ask_grok"
STORE_TYPES = {"user", "feedback", "project", "reference"}


# ---------------------------------------------------------------- reuse existing code
def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


normalize_mod = _load("mp_normalize", REPO / "mempalace" / "normalize.py")
extractor_mod = _load("mp_extractor", REPO / "mempalace" / "general_extractor.py")


# ---------------------------------------------------------------- 4. secret scrub
# Deterministic, pre-provider. Plan §4.6 — never delegated to the LLM.
SECRET_PATTERNS = [
    ("anthropic", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("aws", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("bearer", re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}")),
    ("assign", re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key)\b"
        r"\s*[=:]\s*['\"]?([A-Za-z0-9/_+\-]{12,})['\"]?")),
    # long opaque blob — never needed verbatim in a memory body
    ("blob", re.compile(r"\b[A-Za-z0-9+/=_\-]{40,}\b")),
]


def scrub(text: str) -> tuple[str, int]:
    n = 0
    for label, pat in SECRET_PATTERNS:
        def _sub(m, label=label):
            nonlocal n
            n += 1
            return f"[REDACTED:{label}]"
        text = pat.sub(_sub, text)
    return text, n


# ---------------------------------------------------------------- windowed extraction
# Phase 0 finding: normalize.py drops `thinking` blocks (richest decision source) and
# general_extractor's turn-splitter is too coarse (whole-exchange blobs). This path
# parses the raw jsonl directly, keeps user+assistant text (+ optional thinking),
# excludes tool_use/tool_result, and windows into claim-sized chunks for the LLM.
def extract_segments_raw(path: Path, include_thinking: bool = True) -> list[tuple[str, str]]:
    segs: list[tuple[str, str]] = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = d.get("message")
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        c = m.get("content")
        if isinstance(c, str):
            if role in ("user", "assistant") and c.strip():
                segs.append((role, c.strip()))
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text" and (b.get("text") or "").strip():
                    segs.append((role, b["text"].strip()))
                elif bt == "thinking" and include_thinking:
                    tx = (b.get("thinking") or b.get("text") or "").strip()
                    if tx:
                        segs.append(("thinking", tx))
                # tool_use / tool_result intentionally excluded
    return segs


def window(segs: list[tuple[str, str]], size: int = 1200) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for src, text in segs:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buf = ""
        for p in paras:
            if buf and len(buf) + len(p) + 2 > size:
                chunks.append((src, buf.strip()))
                buf = p
            else:
                buf = f"{buf}\n\n{p}" if buf else p
        if buf.strip():
            chunks.append((src, buf.strip()))
    return [(s, t) for s, t in chunks if len(t) >= 40]


# ---------------------------------------------------------------- existing memory index
def load_existing_index() -> str:
    """name + description lines from current memories (dedup-awareness for the PoC)."""
    lines = []
    for f in sorted(MEM_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        try:
            head = f.read_text(encoding="utf-8", errors="replace")[:600]
        except OSError:
            continue
        name = re.search(r"^name:\s*(.+)$", head, re.M)
        desc = re.search(r"^description:\s*(.+)$", head, re.M)
        if name:
            lines.append(f"- {name.group(1).strip()}: {desc.group(1).strip() if desc else ''}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 5. distill via Grok
DISTILL_SYSTEM = """You extract DURABLE facts about the user Ryan from a Claude Code session, for a
long-term memory store loaded into every FUTURE session. Be RECALL-oriented: extract every fact that
would still be true or useful next week. Durability/dedup filtering happens downstream — your job is
high recall, not gatekeeping.

The store has exactly 4 types:
- user: who Ryan is — role, expertise, durable preferences, identity.
- feedback: how the AI should work for Ryan — corrections, confirmed approaches, workflow rules. Include the WHY.
- project: ongoing work, goals, status, or constraints NOT derivable from code/git.
- reference: pointers to external resources/tools/configs/gotchas (URLs, paths, commands, where-creds-live).

GOOD examples (SHAPE ONLY — these are synthetic; extract the real facts from THIS session, never copy these):
- {"type":"feedback","body":"Run the full test suite before every commit; Ryan treats a red suite as a hard blocker."}
- {"type":"reference","body":"Production logs live in Grafana Loki — query by the service label."}

RULES:
- One fact per memory. Rewrite into a clean, self-contained statement (a fact about Ryan / his work /
  his tools — NOT "the assistant did X just now").
- An EXISTING-MEMORIES list is provided. If a fact is already covered, set "likely_duplicate_of" to that
  name; STILL emit it (downstream decides create-vs-update).
- Bodies are pre-redacted as [REDACTED:*] — never invent or restore secrets.
- Output ONLY a JSON array. No prose. No markdown fences.

Each element:
{"name":"kebab-slug","type":"user|feedback|project|reference","description":"one line",
 "body":"the fact","source_chunk":<int>,"confidence":0.0-1.0,"likely_duplicate_of":"<name or null>"}"""


def grok(system: str, message: str, max_tokens: int = 8000, timeout: float = 180.0) -> str:
    payload = {"message": message, "system": system, "max_tokens": max_tokens}
    req = urllib.request.Request(
        GROK_PROXY, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        body = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        sys.exit(f"grok proxy HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    except urllib.error.URLError as e:
        health = GROK_PROXY.replace("/ask_grok", "/health")
        sys.exit(f"grok proxy unreachable: {getattr(e, 'reason', e)}\n"
                 f"start it: curl -s {health} || "
                 f"(/home/ryan/grok-proxy/.venv/bin/python /home/ryan/grok-proxy/grok_proxy.py &)")
    return body.get("content") or ""


def parse_json_array(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw).rstrip("`").strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


def valid_proposal(p) -> bool:
    return (isinstance(p, dict)
            and isinstance(p.get("name"), str) and p["name"].strip()
            and p.get("type") in STORE_TYPES
            and isinstance(p.get("description"), str)
            and isinstance(p.get("body"), str) and p["body"].strip())


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default=os.environ.get("CLAUDE_CODE_SESSION_ID", ""),
                    help="session uuid or path (default: current session)")
    ap.add_argument("--max-chunks", type=int, default=60, help="cap candidate chunks sent to Grok")
    ap.add_argument("--max-chunk-chars", type=int, default=1500)
    ap.add_argument("--mode", choices=["whole", "windowed", "regex"], default="whole",
                    help="whole = full-session recall-first distill (BEST, Phase 0 finding); "
                         "windowed = per-claim windows (low recall); "
                         "regex = general_extractor prefilter (coarse, for comparison)")
    ap.add_argument("--window", type=int, default=1200, help="windowed mode: chunk size in chars")
    ap.add_argument("--no-thinking", action="store_true", help="windowed mode: exclude thinking blocks")
    args = ap.parse_args()

    tpath = Path(args.transcript)
    if not tpath.exists():
        tpath = MEM_DIR.parent / f"{args.transcript}.jsonl"
    if not tpath.exists():
        sys.exit(f"transcript not found: {args.transcript}")

    print(f"{'='*70}\n  PHASE 0 — Memory Miner PoC  (ZERO WRITES)\n{'='*70}")
    print(f"  Transcript: {tpath.name}  ({tpath.stat().st_size//1024} KB)")

    # 2-3. candidate generation (mode-dependent)
    if args.mode == "whole":
        # Phase 0 finding: full-session context + recall-first prompt >> fragmented chunks.
        # thinking blocks are signature-only (empty) in CC transcripts, so user+assistant text only.
        segs = extract_segments_raw(tpath, include_thinking=False)
        transcript = "\n\n".join(f"{s.upper()}: {t}" for s, t in segs)
        windows = window([("session", transcript)], max(args.window, 12000))
        candidates = [{"content": text, "memory_type": "session"} for _src, text in windows]
        print(f"  Whole-session: {len(segs)} segments → {len(candidates)} window(s), "
              f"thinking unavailable (signature-only), tool noise excluded")
        total_units = len(segs)
        unit_label = "raw segments"
    elif args.mode == "regex":
        text = normalize_mod.normalize(str(tpath))
        if not text or len(text.strip()) < 50:
            sys.exit("normalize produced no usable transcript text")
        print(f"  Normalized: {len(text):,} chars (thinking dropped, coarse turn-split)")
        total_units = len(extractor_mod._split_into_segments(text))
        candidates = [{"content": c["content"], "memory_type": c["memory_type"]}
                      for c in extractor_mod.extract_memories(text)]
        unit_label = "turn-segments"
    else:
        segs = extract_segments_raw(tpath, include_thinking=not args.no_thinking)
        chunks = window(segs, args.window)
        total_units = len(segs)
        candidates = [{"content": t, "memory_type": s} for s, t in chunks]
        nthink = sum(1 for s, _ in segs if s == "thinking")
        print(f"  Raw parse: {len(segs)} segments "
              f"({nthink} thinking{' — excluded' if args.no_thinking else ' — included'}), "
              f"tool noise excluded")
        unit_label = "raw segments"
    if args.max_chunks and len(candidates) > args.max_chunks:
        candidates = candidates[:args.max_chunks]
    src_hist = Counter(c["memory_type"] for c in candidates)
    print(f"\n  -- CANDIDATES ({args.mode} mode) --")
    print(f"  source {unit_label}:  {total_units}")
    print(f"  candidate chunks:    {len(candidates)}")
    print(f"  by source/type:      {dict(src_hist)}")

    if not candidates:
        print("\n  No candidates — nothing to distill. (ZERO files written.)")
        return 0

    # 4. scrub (deterministic, before Grok)
    chunk_cap = max(args.window, 12000) if args.mode == "whole" else args.max_chunk_chars
    scrubbed, redacted = [], 0
    for c in candidates:
        body, n = scrub(c["content"][:chunk_cap])
        redacted += n
        scrubbed.append({**c, "content": body})
    print(f"\n  -- SECRET SCRUB -- redactions: {redacted}")

    # 5. distill via Grok
    existing = load_existing_index()
    numbered = "\n\n".join(
        f"[chunk {i}] (src:{c['memory_type']})\n{c['content']}"
        for i, c in enumerate(scrubbed))
    message = (f"TODAY IS 2026-06-01.\n\nEXISTING MEMORIES (name: description):\n{existing}\n\n"
               f"{'='*40}\nCANDIDATE CHUNKS:\n{numbered}\n\n"
               f"Return the JSON array now.")
    print(f"  -- DISTILL (Grok) -- sending {len(message):,} chars for {len(scrubbed)} chunks ...")
    raw = grok(DISTILL_SYSTEM, message)
    proposals = parse_json_array(raw)
    if proposals is None:
        print("\n  [!] Grok did not return parseable JSON. Raw head:\n")
        print("  " + raw[:800].replace("\n", "\n  "))
        return 1

    valid = [p for p in proposals if valid_proposal(p)]
    dropped = len(proposals) - len(valid)

    # ---- report
    by_type = Counter(p["type"] for p in valid)
    new = [p for p in valid if not p.get("likely_duplicate_of")]
    known = [p for p in valid if p.get("likely_duplicate_of")]
    print(f"\n{'='*70}\n  PROPOSALS  ({len(valid)} valid, {dropped} schema-rejected)\n{'='*70}")
    print(f"  by store type: {dict(by_type)}")
    print(f"  NEW: {len(new)}   |   likely-already-known: {len(known)}\n")

    for p in valid:
        dup = p.get("likely_duplicate_of")
        tag = f"  ~dup of [{dup}]" if dup else "  ** NEW **"
        print(f"  ┌─ [{p['type']}] {p['name']}   conf={p.get('confidence','?')}{tag}")
        print(f"  │  {p['description']}")
        body = p["body"].strip().replace("\n", "\n  │  ")
        print(f"  │  {body}\n  └{'─'*60}")

    # source -> store type overlap (the §5 gate metric)
    print(f"\n  -- SOURCE→STORE TYPE OVERLAP --")
    cross = Counter()
    for p in valid:
        sc = p.get("source_chunk")
        rt = scrubbed[sc]["memory_type"] if isinstance(sc, int) and 0 <= sc < len(scrubbed) else "?"
        cross[(rt, p["type"])] += 1
    for (rt, st), n in sorted(cross.items(), key=lambda x: -x[1]):
        print(f"    regex:{rt:11} -> store:{st:10} {n}")

    print(f"\n{'='*70}\n  ZERO files written. Review above; "
          f"`accept` flow is Phase 1.\n{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
