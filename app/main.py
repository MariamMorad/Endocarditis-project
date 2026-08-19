"""
Application entrypoint.
Starts up FastAPI instantly in 0.1s and builds the RAG index in a background thread,
preventing Azure Container Apps / Ingress 504 gateway timeouts.
Mounts and serves the full EndoAI web frontend directly from Azure.
"""
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

# Mount frontend directory for static assets (images, html)
FRONT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "front")
if os.path.exists(FRONT_DIR):
    app.mount("/front", StaticFiles(directory=FRONT_DIR, html=True), name="front")


@app.get("/", tags=["01 - Health & Meta"], summary="EndoAI Home Web Page")
def root():
    """Serves the interactive EndoAI Landing Page directly on Azure root URL."""
    landing_path = os.path.join(FRONT_DIR, "endoai-landing.html")
    if os.path.exists(landing_path):
        return FileResponse(landing_path)
    return {
        "message": "Clinical RAG API is running.",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "version": "1.0.0",
    }


@app.get("/assistant", tags=["01 - Health & Meta"], summary="EndoAI Assistant Web App")
def assistant():
    """Serves the EndoAI Clinical Assistant directly on Azure."""
    assistant_path = os.path.join(FRONT_DIR, "endoai-assistant.html")
    if os.path.exists(assistant_path):
        return FileResponse(assistant_path)
    return FileResponse(os.path.join(FRONT_DIR, "index.html"))
