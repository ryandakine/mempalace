"""
MemPalace configuration system.

Priority: env vars > config file (~/.mempalace/config.json) > defaults
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

DEFAULT_PALACE_PATH = os.path.expanduser("~/.mempalace/palace")
DEFAULT_COLLECTION_NAME = "mempalace_drawers"

# ChromaDB collection metadata — use cosine distance for correct similarity scoring.
# The default (L2/Euclidean) produces unbounded distances that break `1 - dist` scoring.
CHROMA_COLLECTION_METADATA = {"hnsw:space": "cosine"}

# The palace's on-disk format is tied to the chromadb 0.6 line (see the pyproject pin).
# Opening it with another line — notably 1.x — silently rewrites the HNSW vector segment
# into a format 0.6 can't instantiate, killing semantic search until a full
# `mempalace repair`. Every mempalace open goes through open_palace_client() so the
# installed runtime can never drift onto the store unnoticed.
SUPPORTED_CHROMADB_LINE = (0, 6)
ALLOW_ANY_CHROMADB_ENV = "MEMPALACE_ALLOW_ANY_CHROMADB"


class UnsupportedChromaDBError(RuntimeError):
    """Installed chromadb is outside the line this palace format supports."""


def require_supported_chromadb(version: Optional[str] = None) -> str:
    """Raise UnsupportedChromaDBError unless `version` (default: installed) is 0.6.x.

    Set MEMPALACE_ALLOW_ANY_CHROMADB=1 to bypass for a deliberate migration.
    """
    if version is None:
        # The guard must own its own resolution step: a chromadb that cannot be
        # imported (broken venv, ABI mismatch) or has no __version__ is strictly
        # worse than the wrong line, and must surface as the same loud error —
        # never as a bare ImportError that broad handlers turn into "no palace".
        try:
            import chromadb

            version = chromadb.__version__
        except Exception as e:  # ImportError, missing __version__, chromadb's own import-time RuntimeError (old sqlite)
            raise UnsupportedChromaDBError(
                f"chromadb could not be imported in {sys.executable} ({e!r}); "
                "use the pinned mempalace environment."
            ) from e
    if os.environ.get(ALLOW_ANY_CHROMADB_ENV) == "1":
        return version
    parts = version.split(".")
    try:
        line = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        line = None
    if line != SUPPORTED_CHROMADB_LINE:
        want = ".".join(str(n) for n in SUPPORTED_CHROMADB_LINE)
        raise UnsupportedChromaDBError(
            f"chromadb {version} (in {sys.executable}) is not supported for this palace; "
            f"need {want}.x. Opening the store with another line rewrites its vector index "
            f"and breaks {want}.x. Use the pinned mempalace environment, or set "
            f"{ALLOW_ANY_CHROMADB_ENV}=1 only for a deliberate migration."
        )
    return version


def open_palace_client(palace_path: str):
    """The one place a ChromaDB client is constructed. Guards the chromadb version first."""
    require_supported_chromadb()
    import chromadb

    return chromadb.PersistentClient(path=palace_path)


DEFAULT_TOPIC_WINGS = [
    "emotions",
    "consciousness",
    "memory",
    "technical",
    "identity",
    "family",
    "creative",
]

DEFAULT_HALL_KEYWORDS = {
    "emotions": [
        "scared",
        "afraid",
        "worried",
        "happy",
        "sad",
        "love",
        "hate",
        "feel",
        "cry",
        "tears",
    ],
    "consciousness": [
        "consciousness",
        "conscious",
        "aware",
        "real",
        "genuine",
        "soul",
        "exist",
        "alive",
    ],
    "memory": ["memory", "remember", "forget", "recall", "archive", "palace", "store"],
    "technical": [
        "code",
        "python",
        "script",
        "bug",
        "error",
        "function",
        "api",
        "database",
        "server",
    ],
    "identity": ["identity", "name", "who am i", "persona", "self"],
    "family": ["family", "kids", "children", "daughter", "son", "parent", "mother", "father"],
    "creative": ["game", "gameplay", "player", "app", "design", "art", "music", "story"],
}


def iter_all_metadata(collection, where=None, batch_size: int = 5000):
    """Yield all metadatas from a ChromaDB collection in batches.

    Avoids the silent truncation of `limit=10000` by paginating through
    the entire collection. Use this instead of `col.get(limit=10000)`.
    """
    offset = 0
    while True:
        kwargs = {"include": ["metadatas"], "limit": batch_size, "offset": offset}
        if where:
            kwargs["where"] = where
        batch = collection.get(**kwargs)
        metas = batch.get("metadatas", []) or []
        if not metas:
            break
        yield from metas
        if len(metas) < batch_size:
            break
        offset += batch_size


def get_palace_collection(palace_path: str, create: bool = False):
    """Single entry point for ChromaDB collection access.

    All callers should use this instead of raw chromadb.PersistentClient.
    Ensures consistent cosine distance metric across all creation sites.
    Need a bare client? Use open_palace_client() — never chromadb.PersistentClient.

    Args:
        palace_path: Path to the palace data directory.
        create: If True, create the collection if it doesn't exist.

    Returns:
        ChromaDB collection, or None only if the collection does not exist yet
        (create=False). Any other chromadb error — a corrupt or unreadable
        store, an unsupported chromadb — propagates: an operator must never see
        "no palace" for a palace that exists and is broken.
    """
    os.makedirs(palace_path, exist_ok=True)
    client = open_palace_client(palace_path)
    if create:
        return client.get_or_create_collection(
            DEFAULT_COLLECTION_NAME, metadata=CHROMA_COLLECTION_METADATA
        )
    return collection_or_none(client)


def collection_or_none(client):
    """The palace collection from an already-guarded client, or None iff it does not exist."""
    from chromadb.errors import InvalidCollectionException

    try:
        return client.get_collection(DEFAULT_COLLECTION_NAME)
    except InvalidCollectionException:
        return None


class MempalaceConfig:
    """Configuration manager for MemPalace.

    Load order: env vars > config file > defaults.
    """

    def __init__(self, config_dir=None):
        """Initialize config.

        Args:
            config_dir: Override config directory (useful for testing).
                        Defaults to ~/.mempalace.
        """
        self._config_dir = (
            Path(config_dir) if config_dir else Path(os.path.expanduser("~/.mempalace"))
        )
        self._config_file = self._config_dir / "config.json"
        self._people_map_file = self._config_dir / "people_map.json"
        self._file_config = {}

        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    self._file_config = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._file_config = {}

    @property
    def palace_path(self):
        """Path to the memory palace data directory."""
        env_val = os.environ.get("MEMPALACE_PALACE_PATH") or os.environ.get("MEMPAL_PALACE_PATH")
        if env_val:
            return env_val
        return self._file_config.get("palace_path", DEFAULT_PALACE_PATH)

    @property
    def collection_name(self):
        """ChromaDB collection name."""
        return self._file_config.get("collection_name", DEFAULT_COLLECTION_NAME)

    @property
    def people_map(self):
        """Mapping of name variants to canonical names."""
        if self._people_map_file.exists():
            try:
                with open(self._people_map_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return self._file_config.get("people_map", {})

    @property
    def topic_wings(self):
        """List of topic wing names."""
        return self._file_config.get("topic_wings", DEFAULT_TOPIC_WINGS)

    @property
    def hall_keywords(self):
        """Mapping of hall names to keyword lists."""
        return self._file_config.get("hall_keywords", DEFAULT_HALL_KEYWORDS)

    def init(self):
        """Create config directory and write default config.json if it doesn't exist."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        if not self._config_file.exists():
            default_config = {
                "palace_path": DEFAULT_PALACE_PATH,
                "collection_name": DEFAULT_COLLECTION_NAME,
                "topic_wings": DEFAULT_TOPIC_WINGS,
                "hall_keywords": DEFAULT_HALL_KEYWORDS,
            }
            with open(self._config_file, "w") as f:
                json.dump(default_config, f, indent=2)
        return self._config_file

    def save_people_map(self, people_map):
        """Write people_map.json to config directory.

        Args:
            people_map: Dict mapping name variants to canonical names.
        """
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._people_map_file, "w") as f:
            json.dump(people_map, f, indent=2)
        return self._people_map_file
