"""ChromaDB HTTP client wrapper — connects to the separate chromadb-server container."""

import logging
from typing import Any

import chromadb
from chromadb import Collection

from app.config import get_settings

logger = logging.getLogger(__name__)

_collection: Collection | None = None


def get_collection() -> Collection:
    """Return (or create) the ChromaDB collection. Uses HTTP client for the separate container."""
    global _collection
    if _collection is None:
        settings = get_settings()
        client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
        _collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for legal embeddings
        )
        logger.info(f"Connected to ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}'.")
    return _collection


def add_documents(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """Upsert chunks into ChromaDB. Auto-resets collection if embedding dimensionality changes."""
    if not ids:
        return
    try:
        get_collection().upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.info(f"Upserted {len(ids)} chunks into ChromaDB.")
    except Exception as e:
        if "dimension" in str(e).lower():
            logger.warning(f"Embedding dimension mismatch detected ({e}). Resetting collection for new embedding model...")
            reset_collection()
            get_collection().upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            logger.info(f"Successfully recreated collection and upserted {len(ids)} chunks.")
        else:
            raise


def query_similar(
    embedding: list[float],
    top_k: int = 20,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    """Query ChromaDB by embedding vector. Returns top_k results with metadata and distances."""
    params: dict[str, Any] = {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        params["where"] = where

    raw = get_collection().query(**params)

    results = []
    for i, doc_id in enumerate(raw["ids"][0]):
        results.append({
            "id": doc_id,
            "text": raw["documents"][0][i],
            "metadata": raw["metadatas"][0][i],
            "distance": raw["distances"][0][i],
        })
    return results


def get_chroma_status() -> str:
    """Ping ChromaDB and return 'ok' or an error message."""
    try:
        get_collection()
        return "ok"
    except Exception as e:
        return f"error: {e}"


def reset_collection() -> None:
    """Drop and recreate the collection. Used for re-ingestion."""
    global _collection
    settings = get_settings()
    client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
    client.delete_collection(settings.CHROMA_COLLECTION_NAME)
    _collection = None
    logger.warning(f"Collection '{settings.CHROMA_COLLECTION_NAME}' reset.")
