"""
dedup.py — self-contained, deterministic dedup vs the existing store (plan §4.4).

Rules:
  - Index the FULL store: name + description + BODY of every memory file (the
    ~190 files are tiny; loading them is cheap). The body is where the real
    duplicate signal lives — a proposal ``ask-claude-bridge-live`` and the stored
    ``ask-claude-bridge`` share almost no name tokens but nearly identical bodies.
  - Similarity is a dependency-light SEMANTIC blend, DETERMINISTIC, never
    LLM-judged (Phase 0 §1.5 #4):
        * word TF-IDF cosine  — topical overlap, common words down-weighted
        * char n-gram cosine  — morphological near-duplicates (``-live`` suffix,
                                 hyphen/underscore variants, typos)
    Final score = max(word_cosine, char_cosine), plus an exact name/slug shortcut.
    A plain Jaccard keyword score is kept as a deterministic floor/fallback.
  - score >= HIGH → update; MID → human-review flag; LOW → create.
  - Identity types (user/feedback) NEVER auto-update; a near-duplicate is flagged
    for human review instead (feedback_dont_flip_memory_mid_session).
  - For NON-identity types, when the proposal duplicates an existing memory but
    carries NEW info, the verdict is ``update`` and carries ``enrich=True`` so
    accept.py appends the delta rather than overwriting (merge/enrich path).
  - Generation-time slug-collision check appends -2/-3 (plan §4.4 #4).

Optional embedding booster is supported via an injected ``embed_scorer`` callable
but there is NO hard dependency on MemPalace FTS/embeddings / chromadb /
sentence-transformers (that reintroduces the model-download hang).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .proposal import IDENTITY_TYPES, MemoryProposal, slugify

# Thresholds tuned against the proven-missed cases (see test_dedup.py):
#   - ask-claude-bridge-live vs ask-claude-bridge: char-ngram cosine clears HIGH.
#   - partial "Grok proxy endpoint" restatement lands in [MID, HIGH) → review.
HIGH = 0.55   # >= → update existing (enrich for non-identity, review for identity)
MID = 0.30    # >= → human-review flag

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "is", "are",
    "be", "with", "not", "use", "uses", "via", "by", "at", "it", "his", "her",
    "this", "that", "as", "do", "dont", "don", "t", "ryan", "user", "his",
}

_CHAR_NGRAM = 4  # char n-gram width for the lexical-similarity channel


def _tokens(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOP}


def _token_list(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [w for w in words if len(w) > 2 and w not in _STOP]


def _char_ngrams(text: str, n: int = _CHAR_NGRAM) -> Counter:
    """Char n-grams over a normalised string (alnum + single spaces)."""
    s = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return Counter()
    if len(s) < n:
        return Counter([s])
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _counter_cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class StoreIndex:
    """In-memory index of the existing memory store (plan §4.4 step 1).

    Loads name / description / type frontmatter AND the full body of every
    memory file. Builds a corpus IDF so the word-TF-IDF channel can down-weight
    boilerplate. Self-contained, no chromadb/FTS/sentence-transformers.
    """

    def __init__(self, entries: List[dict]):
        # entry: {file, name, description, type, slug, body, tokens,
        #         token_counts, char_ngrams}
        self.entries = entries
        self._slugs = {e["slug"] for e in entries}
        self._idf = self._build_idf(entries)

    @staticmethod
    def _build_idf(entries: List[dict]) -> Dict[str, float]:
        n = len(entries)
        if n == 0:
            return {}
        df: Counter = Counter()
        for e in entries:
            for w in set(e["token_counts"]):
                df[w] += 1
        # smoothed idf; +1 keeps it strictly positive
        return {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}

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
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            head = text[:2000]
            name = _fm(head, "name")
            desc = _fm(head, "description")
            mtype = _fm(head, "type")
            if not name:
                continue
            body = _strip_frontmatter(text)
            entries.append(_make_entry(f.name, f.stem, name, desc or "", mtype or "", body))
        return cls(entries)

    @property
    def slugs(self) -> set:
        return set(self._slugs)

    def index_lines(self) -> str:
        """`name: description` lines for the distill prompt's existing-memories list."""
        return "\n".join(f"- {e['name']}: {e['description']}" for e in self.entries)

    def _score_pair(self, ptoks: set, p_counts: Counter, p_ngrams: Counter,
                    e: dict) -> float:
        """Deterministic blend: max(word-tfidf-cosine, char-ngram-cosine, jaccard)."""
        word = self._tfidf_cosine(p_counts, e["token_counts"])
        char = _counter_cosine(p_ngrams, e["char_ngrams"])
        jac = _jaccard(ptoks, e["tokens"])
        return max(word, char, jac)

    def _tfidf_cosine(self, a_counts: Counter, b_counts: Counter) -> float:
        if not a_counts or not b_counts:
            return 0.0

        def vec(counts: Counter) -> Dict[str, float]:
            return {w: c * self._idf.get(w, 1.0) for w, c in counts.items()}

        va, vb = vec(a_counts), vec(b_counts)
        common = set(va) & set(vb)
        if not common:
            return 0.0
        dot = sum(va[w] * vb[w] for w in common)
        na = math.sqrt(sum(v * v for v in va.values()))
        nb = math.sqrt(sum(v * v for v in vb.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def best_match(self, proposal: MemoryProposal,
                   embed_scorer: Optional[Callable] = None) -> tuple:
        """Return (best_entry_or_None, score). Deterministic semantic blend over
        name+description+body, optionally boosted by an injected embedding
        scorer (max of the two)."""
        ptext = f"{proposal.name} {proposal.description} {proposal.body}"
        ptoks = _tokens(ptext)
        p_counts = Counter(_token_list(ptext))
        p_ngrams = _char_ngrams(ptext)
        pslug = slugify(proposal.name)
        pname = proposal.name.strip()

        best, best_score = None, 0.0
        for e in self.entries:
            score = self._score_pair(ptoks, p_counts, p_ngrams, e)
            # exact name / slug equality is a strong signal
            if pslug == e["slug"] or pname == e["name"]:
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


def _make_entry(file: str, stem: str, name: str, desc: str, mtype: str, body: str) -> dict:
    blob = f"{name} {desc} {body}"
    return {
        "file": file,
        "name": name,
        "description": desc,
        "type": mtype,
        "slug": stem,  # filename stem is the canonical on-disk slug
        "body": body,
        "tokens": _tokens(blob),
        "token_counts": Counter(_token_list(blob)),
        "char_ngrams": _char_ngrams(blob),
    }


def _strip_frontmatter(text: str) -> str:
    """Return the markdown body, dropping a leading ``---`` frontmatter block."""
    if text.startswith("---"):
        # find the closing fence
        m = re.search(r"^---\s*$.*?^---\s*$", text, re.S | re.M)
        if m:
            return text[m.end():].strip()
    return text.strip()


def _fm(head: str, key: str) -> Optional[str]:
    """Read a frontmatter value. Tolerates BOTH the flat auto-memory format
    (``type: reference`` at column 0) and the nested MemPalace export format
    (``metadata:`` block with an indented ``  type: reference``)."""
    m = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", head, re.M)
    return m.group(1).strip() if m else None


def classify(proposal: MemoryProposal, index: StoreIndex,
             embed_scorer: Optional[Callable] = None) -> Dict:
    """Compute the dedup verdict for one proposal (plan §4.4).

    Returns {action, target, score, slug, enrich} where action ∈
    {create, update, review, skip}.

      - Identity types (user/feedback) NEVER get ``update`` — a near-duplicate is
        routed to ``review`` so a human decides (feedback_dont_flip_memory_mid_session).
      - For NON-identity types at HIGH overlap, ``enrich`` is True so accept.py
        appends the new info onto the existing file instead of duplicating it.
    """
    best, score = index.best_match(proposal, embed_scorer=embed_scorer)
    target = best["file"] if best else None
    is_identity = proposal.type in IDENTITY_TYPES
    enrich = False

    if score >= HIGH:
        if is_identity:
            # Never auto-update identity facts — flag for a human (plan §4.4 #5).
            action = "review"
        else:
            action = "update"
            enrich = True  # merge/enrich the existing file rather than overwrite
    elif score >= MID:
        action = "review"
    else:
        action = "create"
        target = None

    # Reserve a unique slug for create.
    slug = None
    if action == "create":
        slug = index.unique_slug(slugify(proposal.name))

    verdict = {
        "action": action,
        "target": target,
        "score": round(score, 4),
        "slug": slug,
        "enrich": enrich,
    }
    proposal.dedup = {
        "action": action,
        "target": target,
        "score": round(score, 4),
        "enrich": enrich,
    }
    return verdict
