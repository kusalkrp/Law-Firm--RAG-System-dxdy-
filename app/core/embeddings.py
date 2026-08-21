"""Gateway embedding client — wraps openai SDK to call Vercel AI Gateway embeddings."""

import logging
from openai import OpenAI
from app.config import get_settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return a cached OpenAI client pointed at the Vercel AI Gateway."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(
            api_key=settings.AI_GATEWAY_API_KEY,
            base_url=settings.AI_GATEWAY_BASE_URL,
        )
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input string."""
    if not texts:
        return []
    settings = get_settings()
    response = _get_client().embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts,
    )
    logger.info(f"Embedded {len(texts)} texts via Gateway.")
    return [d.embedding for d in response.data]


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
