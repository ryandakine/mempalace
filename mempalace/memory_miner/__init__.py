"""
memory_miner — Phase 1 of the self-evolving memory miner.

Reads NEW Claude Code session transcripts, distills durable facts about the
user, dedups them against the existing auto-memory store, and PROPOSES memory
files. Nothing is written to the live memory store except via the gated
``accept`` command.

See ~/self-evolving-miner-plan.md for the full design (§4 architecture, §6 tests).

Runnable as:  python -m mempalace.memory_miner ...

Phase 1 scope (this package):
  - scan_new()    global record-count watermark + whole-run flock
  - distill()     whole-session, recall-first, synthetic few-shot, pluggable provider
  - secret_scrub() deterministic regex, pre-provider/pre-disk
  - dedup()       deterministic keyword/slug vs existing store files
  - emit()        proposals.jsonl + proposals.md (stable content-hash ids, no dupes)
  - accept        atomic .md write + idempotent MEMORY.md pointer + ledger

Phase 2 (auto-write + self-tune) is intentionally OUT of scope.
"""

from .proposal import MemoryProposal, STORE_TYPES, validate_proposal, proposal_id
from .scrub import secret_scrub, SECRET_PATTERNS

__all__ = [
    "MemoryProposal",
    "STORE_TYPES",
    "validate_proposal",
    "proposal_id",
    "secret_scrub",
    "SECRET_PATTERNS",
]
