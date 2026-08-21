"""BM25 keyword index — sparse retrieval for exact legal terminology matching."""

import logging
import os
import pickle
import re
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_INDEX_PATH = "data/bm25_index.pkl"

# Legal multi-word terms to preserve during tokenization
_LEGAL_PHRASES = [
    "force majeure", "non-compete", "non-solicitation", "governing law",
    "limitation of liability", "intellectual property", "trade secret",
    "hold harmless", "cure period",
]

_bm25: BM25Okapi | None = None
_corpus: list[dict[str, Any]] = []   # parallel list of chunk dicts matching _bm25 index


def _tokenize(text: str) -> list[str]:
    """Lowercase, preserve legal phrases, split on whitespace."""
    text = text.lower()
    for phrase in _LEGAL_PHRASES:
        text = text.replace(phrase, phrase.replace(" ", "_"))
    return re.split(r"\s+", text.strip())


def build_index(chunks: list[dict[str, Any]]) -> None:
    """Build BM25 index from a list of chunk dicts with 'text' and 'metadata' keys."""
    global _bm25, _corpus
    _corpus = chunks
    tokenized = [_tokenize(c["text"]) for c in chunks]
    _bm25 = BM25Okapi(tokenized)

    # Persist to disk so it survives app restarts
    with open(_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": _bm25, "corpus": _corpus}, f)

    logger.info(f"BM25 index built with {len(chunks)} chunks. Saved to {_INDEX_PATH}.")


def load_index() -> bool:
    """Load the persisted BM25 index from disk. Returns True if loaded, False if not found."""
    global _bm25, _corpus
    if not os.path.exists(_INDEX_PATH):
        logger.warning("No BM25 index on disk. Run /ingest first.")
        return False

    with open(_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    _bm25 = data["bm25"]
    _corpus = data["corpus"]
    logger.info(f"BM25 index loaded ({len(_corpus)} chunks).")
    return True


def search(query: str, top_k: int = 20) -> list[dict[str, Any]]:
    """Return top_k chunks ranked by BM25 score against the query."""
    global _bm25, _corpus
    if _bm25 is None:
        if not load_index():
            return []

    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)

    # Pair scores with corpus entries and sort descending
    ranked = sorted(
        [{"score": scores[i], **_corpus[i]} for i in range(len(_corpus))],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[:top_k]
