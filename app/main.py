"""
Application entrypoint.
Starts up FastAPI instantly in 0.1s and builds the RAG index in a background thread,
preventing Azure Container Apps / Ingress 504 gateway timeouts.
"""
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.indexing import rag_index
from app.core.pipeline import ClinicalRAGPipeline
from app.api.routes import router


def _build_rag_background(app: FastAPI):
    print("[startup] Building RAG index in background thread...", flush=True)
    rag_index.build()
    app.state.rag_index = rag_index
    app.state.pipeline = ClinicalRAGPipeline(rag_index)
    print("[startup] Application and RAG Index ready for clinical queries!", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_index = rag_index
    app.state.pipeline = None
    # Start RAG initialization in background thread so server responds immediately
    t = threading.Thread(target=_build_rag_background, args=(app,), daemon=True)
    t.start()
    yield
    print("[shutdown] Shutting down application.", flush=True)


app = FastAPI(
    title="Clinical RAG API — Infective Endocarditis",
    description="RAG pipeline over NICE/ESC guidelines for Infective Endocarditis, "
                "with hybrid retrieval + reranking + agentic evaluation + citation verification.",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "01 - Health & Meta", "description": "Health check and service status endpoints"},
        {"name": "02 - Ingestion", "description": "Guideline indexing and chunk ingestion endpoints"},
        {"name": "03 - Query / RAG", "description": "Clinical question answering and reasoning endpoints"},
        {"name": "04 - Vector Store Admin", "description": "Chroma DB and collection management endpoints"},
    ],
)

# Enable CORS for external/frontend consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["01 - Health & Meta"], summary="Root Health & Documentation Link")
def root():
    return {
        "message": "Clinical RAG API is running.",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "version": "1.0.0",
    }
