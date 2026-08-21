"""Unit and integration tests for the HybridRetriever."""

import pytest
from app.core.retriever import HybridRetriever


def test_hybrid_retrieval_returns_top_k():
    """Verify retriever returns the requested number of top chunks with scores."""
    retriever = HybridRetriever()
    results = retriever.retrieve("governing law and jurisdiction", top_k=5)
    assert len(results) == 5
    assert results[0].rerank_score is not None or results[0].rrf_score > 0
    assert "source_file" in results[0].metadata
    assert "page_number" in results[0].metadata


def test_hybrid_retrieval_clause_filter():
    """Verify filtering by clause_type confines results to matching category."""
    retriever = HybridRetriever()
    results = retriever.retrieve("breach notice", top_k=3, clause_filter="termination")
    assert len(results) > 0
    for r in results:
        assert r.metadata.get("clause_type") == "termination"
