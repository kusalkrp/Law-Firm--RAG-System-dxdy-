"""API route handlers — /health, /query, /ingest."""

import logging
from fastapi import APIRouter
from app.api.schemas import Citation, HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.config import get_settings, validate_gateway_models

logger = logging.getLogger(__name__)
router = APIRouter()


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

    return HealthResponse(
        status="ok" if models_ok else "degraded",
        models_validated=models_ok,
        chromadb_status="pending",  # updated in Phase 3
        models={
            "llm": settings.LLM_MODEL,
            "embedding": settings.EMBEDDING_MODEL,
            "rerank": settings.RERANK_MODEL,
        },
    )


@router.post("/query", response_model=QueryResponse, tags=["RAG Pipeline"])
async def query_contracts(request: QueryRequest) -> QueryResponse:
    """Run the full RAG pipeline and return a cited answer. (stub — wired in Phase 5)"""
    return QueryResponse(
        question=request.question,
        answer="[stub] Full pipeline wired in Phase 5.",
        citations=[Citation(source_file="stub.pdf", page=1, section="1.1", text_excerpt="stub")],
    )


@router.post("/ingest", response_model=IngestResponse, tags=["Document Ingestion"])
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    """Ingest PDFs into vector and BM25 stores. (stub — wired in Phase 3)"""
    return IngestResponse(
        documents_processed=0,
        chunks_created=0,
        status="success",
        message=f"[stub] Directory '{request.directory}' ready.",
    )
