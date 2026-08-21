"""FastAPI app factory, CORS, and startup lifecycle."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings, validate_gateway_models
from app.storage import bm25_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("law_firm_rag")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Validate Gateway models and load persisted BM25 index on startup."""
    settings = get_settings()
    try:
        result = validate_gateway_models(settings)
        logger.info(f"Gateway OK — {result['total_available']} models available.")
    except Exception as e:
        raise RuntimeError(f"Startup failed: {e}") from e

    bm25_index.load_index()  # no-op if index not yet built (pre-first ingest)
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Law Firm RAG System",
        description="Hallucination-resistant RAG API for legal contract analysis.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files and serve web UI at / and /ui
    import os
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")

        @app.get("/", include_in_schema=False)
        @app.get("/ui", include_in_schema=False)
        async def serve_ui() -> FileResponse:
            return FileResponse("static/index.html")

    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
