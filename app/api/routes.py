from fastapi import APIRouter, Request, HTTPException, BackgroundTasks

from app.models.schemas import AskRequest, AskResponse, HealthResponse, CollectionStatsResponse
from app.core.pipeline import ClinicalRAGPipeline
from app.config import settings

router = APIRouter()


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


@router.post("/ask", response_model=AskResponse, tags=["03 - Query / RAG"], summary="Clinical Question Answering")
def ask(payload: AskRequest, request: Request):
    """Executes the Clinical RAG pipeline: Hybrid retrieval -> o4-mini evaluation -> Grounded generation -> Citation verification."""
    pipeline = get_pipeline(request)
    result = pipeline.answer(payload.question, k=payload.k)
    return result


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
