import os
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.models.schemas import AskRequest, AskResponse, HealthResponse, CollectionStatsResponse
from app.core.pipeline import ClinicalRAGPipeline
from app.core.prompt_refiner import refine_user_prompt, RefinedPromptResult
from app.config import settings

router = APIRouter()


class PromptRefineRequest(BaseModel):
    query: str = Field(..., example="tooth brushing and endocarditis")


def get_pipeline(request: Request) -> ClinicalRAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None or not getattr(request.app.state.rag_index, "ready", False):
        raise HTTPException(status_code=503, detail="Index is not ready yet. Try again shortly or call /reindex.")
    return pipeline


@router.get("/health", response_model=HealthResponse, tags=["01 - Health & Meta"], summary="Health & Index Status")
def health(request: Request):
    """Returns server health and index readiness status."""
    index = getattr(request.app.state, "rag_index", None)
    ready = bool(index and index.ready)
    total_chunks = len(index.langchain_docs) if ready else 0
    return HealthResponse(status="ok", index_ready=ready, total_chunks=total_chunks)


@router.post("/reindex", tags=["02 - Ingestion"], summary="Rebuild RAG Index")
def reindex(request: Request, background_tasks: BackgroundTasks):
    """Rebuilds the index (extract -> chunk -> dedup -> embed -> BM25 -> reranker) in the background."""
    def _rebuild() -> None:
        request.app.state.rag_index.build()
        request.app.state.pipeline = ClinicalRAGPipeline(request.app.state.rag_index)

    background_tasks.add_task(_rebuild)
    return {"status": "reindex_started"}


@router.get("/documents", tags=["02 - Ingestion"], summary="List Available Medical Guidelines")
def list_documents():
    """Lists the clinical guideline PDFs currently available in the server."""
    docs = []
    for filename in settings.pdf_list:
        path = os.path.join(settings.PDF_DIR, filename)
        if os.path.exists(path):
            docs.append({
                "filename": filename,
                "size_bytes": os.path.getsize(path),
                "download_url": f"/api/v1/documents/{filename}",
                "guideline": "ESC Guidelines 2023" if "ESC" in filename else "NICE Guidelines"
            })
    return {"documents": docs}


@router.get("/documents/{filename}", tags=["02 - Ingestion"], summary="Download/View Raw Medical Guideline PDF")
def get_document(filename: str):
    """Serves the exact clinical guideline PDF file for in-browser inspection."""
    if filename not in settings.pdf_list:
        raise HTTPException(status_code=404, detail="Requested guideline PDF not found.")
    path = os.path.join(settings.PDF_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="PDF file missing on server.")
    return FileResponse(path, media_type="application/pdf", filename=filename)


@router.post("/ask", response_model=AskResponse, tags=["03 - Query / RAG"], summary="Clinical Question Answering")
def ask(payload: AskRequest, request: Request):
    """Executes the Clinical RAG pipeline: Hybrid retrieval -> o4-mini evaluation -> Grounded generation -> Citation verification."""
    pipeline = get_pipeline(request)
    result = pipeline.answer(payload.question, k=payload.k)
    return result


@router.post("/refine-prompt", response_model=RefinedPromptResult, tags=["03 - Query / RAG"], summary="AI Prompt Refiner & Optimizer")
def refine_prompt_endpoint(payload: PromptRefineRequest, request: Request):
    """Refines rough clinical queries into chunk-grounded prompts. Uses actual index sections to guarantee evidence retrieval."""
    rag_index = getattr(request.app.state, "rag_index", None)
    return refine_user_prompt(payload.query, rag_index=rag_index)


@router.get("/collections/stats", response_model=CollectionStatsResponse, tags=["04 - Vector Store Admin"], summary="Vector Store & Collection Info")
def collection_stats(request: Request):
    """Returns collection statistics and metadata from the Chroma vector store."""
    index = getattr(request.app.state, "rag_index", None)
    ready = bool(index and index.ready)
    total_chunks = len(index.langchain_docs) if ready else 0
    return CollectionStatsResponse(
        collection_name=settings.CHROMA_COLLECTION,
        total_chunks=total_chunks,
        embedding_model=settings.EMBEDDING_MODEL,
        reranker_model=settings.RERANKER_MODEL,
        persist_dir=settings.CHROMA_PERSIST_DIR,
    )
