"""API route handlers — /health, /query, /ingest."""

import logging
import time
from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    Citation,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.config import get_settings, validate_gateway_models
from app.core import embeddings
from app.core.document_processor import DocumentProcessor
from app.core.evaluator import FaithfulnessEvaluator
from app.core.generator import ResponseGenerator
from app.core.retriever import HybridRetriever
from app.storage import bm25_index, vector_store

logger = logging.getLogger(__name__)
router = APIRouter()

_retriever = HybridRetriever()
_generator = ResponseGenerator()
_evaluator = FaithfulnessEvaluator()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Gateway model check + ChromaDB ping. Re-runs on every call for container liveness."""
    settings = get_settings()

    try:
        validate_gateway_models(settings)
        models_ok = True
    except Exception as e:
        logger.warning(f"Model validation failed: {e}")
        models_ok = False

    chroma_status = vector_store.get_chroma_status()

    return HealthResponse(
        status="ok" if models_ok and chroma_status == "ok" else "degraded",
        models_validated=models_ok,
        chromadb_status=chroma_status,
        models={
            "llm": settings.LLM_MODEL,
            "embedding": settings.EMBEDDING_MODEL,
            "rerank": settings.RERANK_MODEL,
        },
    )


@router.post("/query", response_model=QueryResponse, tags=["RAG Pipeline"])
async def query_contracts(request: QueryRequest) -> QueryResponse:
    """End-to-end legal query: Hybrid Retrieval -> LLM Generation -> Faithfulness Evaluation."""
    start_time = time.perf_counter()

    try:
        # 1. Retrieve top-k relevant chunks (Dense + Sparse + RRF + Cohere Rerank)
        chunks = _retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            clause_filter=request.clause_filter,
        )

        # 2. Generate grounded answer with citations and hallucination guards
        generated = _generator.generate(query=request.question, chunks=chunks)

        # 3. Evaluate claim-level faithfulness against the source passages
        faithfulness = _evaluator.evaluate(answer=generated.answer, chunks=chunks)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return QueryResponse(
            question=request.question,
            answer=generated.answer,
            citations=generated.citations,
            faithfulness_score=faithfulness.score,
            confidence=generated.confidence,
            abstention_triggered_by=generated.abstention_triggered_by,
            processing_time_ms=latency_ms,
        )
    except Exception as e:
        logger.error(f"Query pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@router.post("/ingest", response_model=IngestResponse, tags=["Document Ingestion"])
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    """Parse PDFs -> embed -> store in ChromaDB + build BM25 index."""
    settings = get_settings()
    processor = DocumentProcessor(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    try:
        chunks = processor.process_directory(request.directory)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not chunks:
        return IngestResponse(
            documents_processed=0,
            chunks_created=0,
            status="success",
            message="No PDF files found in directory.",
        )

    texts = [c.text for c in chunks]
    try:
        chunk_embeddings = embeddings.embed_texts(texts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}")

    try:
        vector_store.add_documents(
            ids=[c.chunk_id for c in chunks],
            embeddings=chunk_embeddings,
            documents=texts,
            metadatas=[c.metadata for c in chunks],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ChromaDB upsert failed: {e}")

    bm25_corpus = [{"text": c.text, "metadata": c.metadata, "id": c.chunk_id} for c in chunks]
    bm25_index.build_index(bm25_corpus)

    doc_names = list({c.source_file for c in chunks})

    return IngestResponse(
        documents_processed=len(doc_names),
        chunks_created=len(chunks),
        status="success",
        message=f"Ingested {len(doc_names)} documents ({len(chunks)} chunks).",
    )
