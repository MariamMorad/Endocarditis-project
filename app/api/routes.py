from fastapi import APIRouter, Request, HTTPException, BackgroundTasks

from app.models.schemas import AskRequest, AskResponse, HealthResponse
from app.core.pipeline import ClinicalRAGPipeline

router = APIRouter()


def get_pipeline(request: Request) -> ClinicalRAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None or not request.app.state.rag_index.ready:
        raise HTTPException(status_code=503, detail="Index is not ready yet. Try again shortly or call /reindex.")
    return pipeline


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    index = getattr(request.app.state, "rag_index", None)
    ready = bool(index and index.ready)
    total_chunks = len(index.langchain_docs) if ready else 0
    return HealthResponse(status="ok", index_ready=ready, total_chunks=total_chunks)


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request):
    pipeline = get_pipeline(request)
    result = pipeline.answer(payload.question, k=payload.k)
    return result


def _rebuild_index(request: Request) -> None:
    request.app.state.rag_index.build()
    request.app.state.pipeline = ClinicalRAGPipeline(request.app.state.rag_index)


@router.post("/reindex")
def reindex(request: Request, background_tasks: BackgroundTasks):
    """Rebuilds the index (extract -> chunk -> dedup -> embed -> BM25 -> reranker) in the background."""
    background_tasks.add_task(_rebuild_index, request)
    return {"status": "reindex_started"}
