"""
dedup.py — self-contained, deterministic dedup vs the existing store (plan §4.4).

Rules:
  - Always: keyword/slug overlap vs the existing files' name:/description: lines.
    Cheap, no deps, DETERMINISTIC — never LLM-judged (Phase 0 §1.5 #4).
  - score >= HIGH → update; MID → human-review flag; LOW → create.
  - Identity types (user/feedback) NEVER auto-update; a near-duplicate is flagged
    for human review instead (feedback_dont_flip_memory_mid_session).
  - Generation-time slug-collision check appends -2/-3 (plan §4.4 #4).

Optional embedding booster is supported via an injected ``embed_scorer`` callable
but there is NO hard dependency on MemPalace FTS/embeddings.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .proposal import IDENTITY_TYPES, MemoryProposal, slugify

HIGH = 0.62   # >= → update existing
MID = 0.42    # >= → human-review flag

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "is", "are",
    "be", "with", "not", "use", "uses", "via", "by", "at", "it", "his", "her",
    "this", "that", "as", "do", "dont", "don", "t", "ryan", "user", "his",
}


def _tokens(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOP}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class StoreIndex:
    """In-memory index of the existing memory store (plan §4.4 step 1).

    Reads only frontmatter ``name:``/``description:``/``type:`` lines and the
    filename — never loads full bodies. Self-contained, no chromadb/FTS.
    """

    def __init__(self, entries: List[dict]):
        # entry: {file, name, description, type, slug, tokens}
        self.entries = entries
        self._slugs = {e["slug"] for e in entries}

    @classmethod
    def from_dir(cls, memory_dir: Path) -> "StoreIndex":
        memory_dir = Path(memory_dir)
        entries: List[dict] = []
        if not memory_dir.exists():
            return cls(entries)
        for f in sorted(memory_dir.glob("*.md")):
            if f.name == "MEMORY.md":
                continue
            try:
                head = f.read_text(encoding="utf-8", errors="replace")[:1200]
            except OSError:
                continue
            name = _fm(head, "name")
            desc = _fm(head, "description")
            mtype = _fm(head, "type")
            if not name:
                continue
            entries.append(
                {
                    "file": f.name,
                    "name": name,
                    "description": desc or "",
                    "type": mtype or "",
                    "slug": f.stem,  # filename stem is the canonical on-disk slug
                    "tokens": _tokens(f"{name} {desc}"),
                }
            )
        return cls(entries)

    @property
    def slugs(self) -> set:
        return set(self._slugs)

    def index_lines(self) -> str:
        """`name: description` lines for the distill prompt's existing-memories list."""
        return "\n".join(f"- {e['name']}: {e['description']}" for e in self.entries)

    def best_match(self, proposal: MemoryProposal,
                   embed_scorer: Optional[Callable] = None) -> tuple:
        """Return (best_entry_or_None, score). Deterministic keyword overlap,
        optionally boosted by an injected embedding scorer (max of the two)."""
        ptoks = _tokens(f"{proposal.name} {proposal.description} {proposal.body}")
        best, best_score = None, 0.0
        for e in self.entries:
            score = _jaccard(ptoks, e["tokens"])
            # slug equality is a strong signal
            if slugify(proposal.name) == e["slug"] or proposal.name.strip() == e["name"]:
                score = max(score, 0.95)
            if embed_scorer is not None:
                try:
                    score = max(score, float(embed_scorer(proposal, e)))
                except Exception:
                    pass
            if score > best_score:
                best, best_score = e, score
        return best, best_score

    def unique_slug(self, base_slug: str) -> str:
        """Append -2/-3/... until the slug is free on disk (plan §4.4 #4)."""
        slug = base_slug
        if slug not in self._slugs:
            self._slugs.add(slug)
            return slug
        i = 2
        while f"{base_slug}_{i}" in self._slugs:
            i += 1
        slug = f"{base_slug}_{i}"
        self._slugs.add(slug)
        return slug


def _fm(head: str, key: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", head, re.M)
    return m.group(1).strip() if m else None


def classify(proposal: MemoryProposal, index: StoreIndex,
             embed_scorer: Optional[Callable] = None) -> Dict:
    """Compute the dedup verdict for one proposal (plan §4.4).

    Returns {action, target, score, slug} where action ∈
    {create, update, review, skip}. Identity types never get ``update``.
    """
    best, score = index.best_match(proposal, embed_scorer=embed_scorer)
    target = best["file"] if best else None
    is_identity = proposal.type in IDENTITY_TYPES

    if score >= HIGH:
        if is_identity:
            # Never auto-update identity facts — flag for a human (plan §4.4 #5).
            action = "review"
        else:
            action = "update"
    elif score >= MID:
        action = "review"
    else:
        action = "create"
        target = None

    # Reserve a unique slug for create/review-that-becomes-create.
    slug = None
    if action == "create":
        slug = index.unique_slug(slugify(proposal.name))

    verdict = {"action": action, "target": target, "score": round(score, 4), "slug": slug}
    proposal.dedup = {"action": action, "target": target, "score": round(score, 4)}
    return verdict
