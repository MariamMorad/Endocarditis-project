"""
Central application settings, loaded from a .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ---- Gemini ----
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # ---- PDFs ----
    PDF_DIR: str = "data/PDFs"
    PDF_FILES: str = "ESC.pdf,NICE.pdf"

    # ---- Chunking ----
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 150

    # ---- Embeddings / Reranker ----
    EMBEDDING_MODEL: str = "NeuML/pubmedbert-base-embeddings"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- Vector store ----
    CHROMA_COLLECTION: str = "clinical_guidelines_v2"

    @property
    def pdf_paths(self) -> list[str]:
        return [f"{self.PDF_DIR.rstrip('/')}/{name.strip()}" for name in self.PDF_FILES.split(",")]


settings = Settings()
