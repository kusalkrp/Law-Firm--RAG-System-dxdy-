"""Hybrid retrieval engine — combines dense vector search, BM25 sparse search, RRF fusion, and Cohere reranking."""

import logging
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core import embeddings
from app.core.document_processor import DocumentProcessor
from app.storage import bm25_index, vector_store

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    """A retrieved chunk candidate with metadata, fusion score, and rerank score."""
    id: str
    text: str
    metadata: Dict[str, Any]
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None


class HybridRetriever:
    """Combines dense semantic retrieval and sparse BM25 retrieval with RRF and Cohere rerank."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.processor = DocumentProcessor()

    def _dense_search(self, query: str, top_k: int, clause_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Dense similarity search via ChromaDB vector store."""
        query_vec = embeddings.embed_query(query)
        where = {"clause_type": clause_filter} if clause_filter else None
        return vector_store.query_similar(embedding=query_vec, top_k=top_k, where=where)

    def _sparse_search(self, query: str, top_k: int, clause_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Sparse keyword search via BM25."""
        results = bm25_index.search(query=query, top_k=top_k * 2 if clause_filter else top_k)
        if clause_filter:
            results = [r for r in results if r.get("metadata", {}).get("clause_type") == clause_filter][:top_k]
        return results

    def _apply_rrf(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[RetrievedChunk]:
        """Merge dense and sparse results using Reciprocal Rank Fusion (RRF)."""
        candidates: Dict[str, RetrievedChunk] = {}

        # Process dense rankings
        for rank, item in enumerate(dense_results, start=1):
            doc_id = item["id"]
            rrf_val = 1.0 / (k + rank)
            candidates[doc_id] = RetrievedChunk(
                id=doc_id,
                text=item["text"],
                metadata=item["metadata"],
                dense_rank=rank,
                rrf_score=rrf_val,
            )

        # Process sparse rankings
        for rank, item in enumerate(sparse_results, start=1):
            doc_id = item["id"]
            rrf_val = 1.0 / (k + rank)
            if doc_id in candidates:
                candidates[doc_id].sparse_rank = rank
                candidates[doc_id].rrf_score += rrf_val
            else:
                candidates[doc_id] = RetrievedChunk(
                    id=doc_id,
                    text=item["text"],
                    metadata=item["metadata"],
                    sparse_rank=rank,
                    rrf_score=rrf_val,
                )

        # Sort by combined RRF score descending
        fused = sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)
        return fused

    def _rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
        """Rerank candidates using Cohere rerank via the Vercel AI Gateway."""
        if not chunks:
            return []

        doc_texts = [c.text for c in chunks]
        url = f"{self.settings.AI_GATEWAY_BASE_URL.rstrip('/')}/rerank"
        headers = {
            "Authorization": f"Bearer {self.settings.AI_GATEWAY_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.RERANK_MODEL,
            "query": query,
            "documents": doc_texts,
            "top_n": min(top_n, len(chunks)),
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(url, json=payload, headers=headers)
                res.raise_for_status()
                data = res.json()

            reranked_results = []
            for item in data.get("results", []):
                idx = item["index"]
                score = item["relevance_score"]
                chunk = chunks[idx]
                chunk.rerank_score = score
                reranked_results.append(chunk)

            return reranked_results
        except Exception as e:
            logger.warning(f"Rerank call failed ({e}); falling back to RRF rank order.")
            return chunks[:top_n]

    VALID_CLAUSE_TYPES = {
        "indemnification", "termination", "governing_law", "limitation_of_liability",
        "confidentiality", "intellectual_property", "force_majeure", "payment_terms",
        "non_compete", "warranties", "general",
    }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        clause_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Execute the full hybrid retrieval pipeline: Dense + Sparse -> RRF -> Cohere Rerank."""
        fusion_k = self.settings.FUSION_TOP_K
        rrf_k = self.settings.RRF_K

        # Sanitize clause filter (ignore Swagger placeholder 'string' or unknown types)
        effective_filter = clause_filter if clause_filter in self.VALID_CLAUSE_TYPES else None

        # 1. Run dense and sparse searches in parallel candidate pools
        dense_hits = self._dense_search(query, top_k=fusion_k, clause_filter=effective_filter)
        # If hard filter returned too few results, fall back to unrestricted dense search
        if effective_filter and len(dense_hits) < 3:
            unrestricted_dense = self._dense_search(query, top_k=fusion_k, clause_filter=None)
            existing_ids = {h.chunk_id for h in dense_hits}
            for h in unrestricted_dense:
                if h.chunk_id not in existing_ids:
                    dense_hits.append(h)

        sparse_hits = self._sparse_search(query, top_k=fusion_k, clause_filter=effective_filter)
        if effective_filter and len(sparse_hits) < 3:
            unrestricted_sparse = self._sparse_search(query, top_k=fusion_k, clause_filter=None)
            existing_ids = {h.chunk_id for h in sparse_hits}
            for h in unrestricted_sparse:
                if h.chunk_id not in existing_ids:
                    sparse_hits.append(h)

        # 2. Fuse candidate pools via Reciprocal Rank Fusion
        fused_candidates = self._apply_rrf(dense_hits, sparse_hits, k=rrf_k)[:fusion_k]

        # 3. Rerank the top fused candidates with cross-encoder
        final_top = self._rerank(query, fused_candidates, top_n=top_k)

        logger.info(f"Retrieved {len(final_top)} chunks for query: '{query[:50]}...'")
        return final_top
