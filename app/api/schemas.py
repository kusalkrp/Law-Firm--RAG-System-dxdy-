"""Request/response schemas for all API endpoints."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single clause citation from a source contract."""
    source_file: str
    page: int
    section: str
    clause_type: str = "general"
    text_excerpt: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    clause_filter: Optional[str] = None  # e.g. "termination", "indemnification"


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation] = []
    faithfulness_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: Literal["high", "medium", "low"] = "high"
    abstention_triggered_by: Optional[Literal["pre_check", "model"]] = None
    processing_time_ms: Optional[float] = None


class IngestRequest(BaseModel):
    directory: str = "data/contracts"


class IngestResponse(BaseModel):
    documents_processed: int
    chunks_created: int
    status: Literal["success", "partial", "failed"]
    message: str


class HealthResponse(BaseModel):
    status: str                  # "ok" or "degraded"
    models_validated: bool
    chromadb_status: str
    models: dict
