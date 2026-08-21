"""App configuration — loaded from .env file."""

import logging
from functools import lru_cache
from typing import Any, List, Union

import httpx
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Gateway — required, no default (must be set in .env)
    AI_GATEWAY_API_KEY: str
    AI_GATEWAY_BASE_URL: str = "https://ai-gateway.vercel.sh/v1"

    # Models
    LLM_MODEL: str = "anthropic/claude-sonnet-4.6"
    EMBEDDING_MODEL: str = "openai/text-embedding-3-large"
    RERANK_MODEL: str = "cohere/rerank-v3.5"

    # ChromaDB
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "legal_contracts"

    # Retrieval & chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 75
    RRF_K: int = 60
    FUSION_TOP_K: int = 20
    RERANK_TOP_N: int = 5

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: Union[List[str], str] = ["*"]
    DEBUG: bool = False

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v: Any) -> List[str]:
        """Accept comma-separated string or list."""
        if isinstance(v, str):
            return ["*"] if v.strip() == "*" else [o.strip() for o in v.split(",") if o.strip()]
        return v or ["*"]


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


def validate_gateway_models(settings: Settings | None = None) -> dict:
    """Check that configured model IDs exist on the Gateway. Raises RuntimeError if any are missing."""
    cfg = settings or get_settings()

    try:
        resp = httpx.get(
            f"{cfg.AI_GATEWAY_BASE_URL.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {cfg.AI_GATEWAY_API_KEY}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        available = {m["id"] for m in resp.json().get("data", [])}
    except Exception as e:
        raise RuntimeError(f"Could not reach Vercel AI Gateway: {e}") from e

    required = {
        "LLM_MODEL": cfg.LLM_MODEL,
        "EMBEDDING_MODEL": cfg.EMBEDDING_MODEL,
        "RERANK_MODEL": cfg.RERANK_MODEL,
    }

    missing = {k: v for k, v in required.items() if v not in available}
    if missing:
        raise RuntimeError(
            f"Models not found on Gateway: {missing}. Update .env with valid model IDs."
        )

    for v in required.values():
        logger.info(f"✓ Model validated: {v}")

    return {"status": "valid", "validated_models": required, "total_available": len(available)}
