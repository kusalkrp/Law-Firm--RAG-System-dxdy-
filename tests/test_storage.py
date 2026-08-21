"""Tests for storage layer: BM25 index and ChromaDB vector store."""

import pytest
from app.core.document_processor import DocumentProcessor
from app.core import embeddings
from app.storage import bm25_index, vector_store


def test_bm25_index():
    """Verify BM25 index creation, search, and disk persistence."""
    processor = DocumentProcessor()
    chunks = processor.process_directory("data/contracts")
    assert len(chunks) > 0, "No chunks extracted from data/contracts"

    corpus = [{"text": c.text, "metadata": c.metadata, "id": c.chunk_id} for c in chunks]
    bm25_index.build_index(corpus)

    results = bm25_index.search("termination notice period", top_k=5)
    assert len(results) == 5
    assert results[0]["score"] > 0

    # Persistence check
    bm25_index._bm25 = None
    bm25_index._corpus = []
    assert bm25_index.load_index() is True
    assert len(bm25_index.search("indemnification liability", top_k=3)) == 3


def test_chroma_vector_store():
    """Verify ChromaDB connectivity, upsert, and vector query."""
    status = vector_store.get_chroma_status()
    assert status == "ok", f"ChromaDB not healthy: {status}"

    # Embed and query
    q_vec = embeddings.embed_query("What are the governing law and jurisdiction terms?")
    assert len(q_vec) > 0

    results = vector_store.query_similar(q_vec, top_k=5)
    assert len(results) == 5
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "distance" in results[0]
    assert results[0]["distance"] >= 0.0
