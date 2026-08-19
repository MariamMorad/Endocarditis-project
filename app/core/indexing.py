"""
Build the index: Chroma vector store + BM25 retriever + Cross-Encoder reranker.
Everything lives inside one class (RAGIndex) instead of module-level globals, so we can:
  - safely build/rebuild it from a /reindex endpoint
  - store it on app.state and inject it via Depends into routes
"""
import os
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from app.config import settings
from app.core.pdf_processing import extract_all_pdfs
from app.core.chunking import chunk_merged_sections
from app.core.dedup import deduplicate_chunks


@dataclass
class RAGIndex:
    langchain_docs: list[Document] = field(default_factory=list)
    vector_store: Chroma | None = None
    bm25_retriever: BM25Retriever | None = None
    vector_retriever = None
    reranker: CrossEncoder | None = None
    embeddings: HuggingFaceEmbeddings | None = None
    ready: bool = False

    def build(self) -> None:
        print("[RAGIndex] Extracting PDFs...", flush=True)
        all_documents = extract_all_pdfs(settings.pdf_paths)

        print("[RAGIndex] Chunking...", flush=True)
        rag_chunks = chunk_merged_sections(
            all_documents,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        print("[RAGIndex] Deduplicating...", flush=True)
        rag_chunks = deduplicate_chunks(rag_chunks)

        self.langchain_docs = [
            Document(
                page_content=chunk["text"],
                metadata={
                    "source": chunk["source"],
                    "section": chunk["section"],
                    "section_start_page": chunk["section_start_page"],
                    "chunk_id": chunk["chunk_id"],
                },
            )
            for chunk in rag_chunks
        ]
        print(f"[RAGIndex] Loaded {len(self.langchain_docs)} deduplicated chunks.", flush=True)

        print("[RAGIndex] Loading embedding model (PubMedBERT)...", flush=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

        print("[RAGIndex] Creating/Loading persistent vector database...", flush=True)
        persist_dir = settings.CHROMA_PERSIST_DIR
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self.vector_store = Chroma.from_documents(
                documents=self.langchain_docs,
                embedding=self.embeddings,
                collection_name=settings.CHROMA_COLLECTION,
                collection_metadata={"hnsw:space": "cosine"},
                persist_directory=persist_dir,
            )
        else:
            self.vector_store = Chroma.from_documents(
                documents=self.langchain_docs,
                embedding=self.embeddings,
                collection_name=settings.CHROMA_COLLECTION,
                collection_metadata={"hnsw:space": "cosine"},
            )

        print("[RAGIndex] Initializing BM25 retriever (tuned k1/b)...", flush=True)
        self.bm25_retriever = BM25Retriever.from_documents(
            self.langchain_docs,
            bm25_params={"k1": 1.8, "b": 0.85},
        )
        self.bm25_retriever.k = 20

        print("[RAGIndex] Initializing similarity-based vector retriever...", flush=True)
        self.vector_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20},
        )

        print("[RAGIndex] Loading cross-encoder reranker...", flush=True)
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)

        self.ready = True
        print("[RAGIndex] Pipeline ready.", flush=True)

    def get_similarity_scores(self, query: str, k: int, filter_: dict | None = None):
        """Cosine distance -> bounded similarity, computed manually instead of relying
        on Chroma's default relevance_score_fn."""
        results = self.vector_store.similarity_search_with_score(query, k=k, filter=filter_)
        return [(doc, 1 - (dist / 2)) for doc, dist in results]


# A single instance, built once at startup and stored on app.state
rag_index = RAGIndex()
