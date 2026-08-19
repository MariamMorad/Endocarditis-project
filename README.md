# Clinical Decision Support RAG API — Infective Endocarditis
Live Demo: EndoAI Clinical AI Assistant

A production-grade Clinical Retrieval-Augmented Generation (RAG) API built over NICE and ESC guidelines for Infective Endocarditis.

## Architecture Highlights
- **Hybrid Retrieval:** Dense Vector Search (Chroma DB) + Sparse Keyword Search (BM25) with reciprocal rank score fusion.
- **Reranker:** Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for candidate re-scoring.
- **Agentic QC Evaluator:** Evaluates question scope and evidence sufficiency using Azure OpenAI (`o4-mini`).
- **Grounded Generator:** Generates structured recommendations strictly constrained to retrieved guideline evidence.
- **Citation Verification & Repair:** Substring and longest-match coverage verification to prevent hallucinated chunk IDs and fabricated quotes.

---

## 1. Quickstart (Local Development)

### 1.1 Environment Setup
Copy [.env.example](file:///c:/Users/ahmed/.gemini/antigravity-ide/scratch/Endocarditis-project/.env.example) to `.env`:
```env
OPENAI_API_KEY=your_azure_openai_api_key_here
OPENAI_BASE_URL=https://ah30309142502238-8748-resource.openai.azure.com/openai/v1
OPENAI_MODEL=o4-mini

PDF_DIR=data/PDFs
PDF_FILES=ESC.pdf,NICE.pdf
CHUNK_SIZE=900
CHUNK_OVERLAP=150
EMBEDDING_MODEL=NeuML/pubmedbert-base-embeddings
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CHROMA_COLLECTION=clinical_guidelines_v2
CHROMA_PERSIST_DIR=data/chroma_db
```

### 1.2 Install & Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI Specification: `http://127.0.0.1:8000/openapi.json`

---

## 2. Apidog Integration

### 2.1 Importing Endpoints into Apidog
1. Open Apidog and click **`+`** $\rightarrow$ **Import Data** $\rightarrow$ **OpenAPI / Swagger**.
2. Select **Import from URL** and enter: `http://localhost:8000/openapi.json`.
3. Check **Group by Tags** to automatically get the structured folders:
   - `01 - Health & Meta` (`GET /`, `GET /api/v1/health`)
   - `02 - Ingestion` (`POST /api/v1/reindex`)
   - `03 - Query / RAG` (`POST /api/v1/ask`)
   - `04 - Vector Store Admin` (`GET /api/v1/collections/stats`)

### 2.2 Environment Variables in Apidog
| Variable | `local` | `staging (Azure)` | `production (Azure)` |
|---|---|---|---|
| `base_url` | `http://localhost:8000` | `https://<staging-app>.azurecontainerapps.io` | `https://<prod-app>.azurecontainerapps.io` |
| `default_k` | `8` | `8` | `8` |

---

## 3. Azure Deployment (Azure Container Apps)

### Automated One-Click Deployment
- **Bash (Linux/Mac):** `./deploy_azure.sh`
- **PowerShell (Windows):** `.\deploy_azure.ps1`

### Deployment Features
1. Multi-stage Docker image built and pushed to **Azure Container Registry (ACR)**.
2. Secrets securely stored in **Azure Key Vault**.
3. Chroma DB persisted to an **Azure Files Share** mounted directly into Azure Container Apps.
4. CI/CD automated via [.github/workflows/deploy.yml](file:///c:/Us
ers/ahmed/.gemini/antigravity-ide/scratch/Endocarditis-project/.github/workflows/deploy.yml).
