"""
Application entrypoint. On server startup (lifespan), we:
  1. Read PDFs -> extract -> chunk -> dedup
  2. Build Chroma vector store + BM25 + Cross-Encoder reranker
  3. Prepare ClinicalRAGPipeline and store it on app.state
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.indexing import rag_index
from app.core.pipeline import ClinicalRAGPipeline
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Building RAG index...")
    rag_index.build()
    app.state.rag_index = rag_index
    app.state.pipeline = ClinicalRAGPipeline(rag_index)
    print("[startup] Application ready.")
    yield
    print("[shutdown] Shutting down application.")


app = FastAPI(
    title="Clinical RAG API — Infective Endocarditis",
    description="RAG pipeline over NICE/ESC guidelines for Infective Endocarditis, "
                "with hybrid retrieval + reranking + agentic evaluation + citation verification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1", tags=["clinical-rag"])


@app.get("/")
def root():
    return {"message": "Clinical RAG API is running. Go to /docs to try the endpoints."}
