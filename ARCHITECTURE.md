# 🫀 EndoAI: Clinical RAG Architecture & Technical Specification

> **A Clinically Grounded Medical Question-Answering System for Infective Endocarditis Guidelines (ESC 2023 & NICE CG64)**  
> *Powered by Hybrid Retrieval (PubMedBERT + BM25), Cross-Encoder Reranking, Azure OpenAI `o4-mini`, and Zero-Hallucination Citation Verification.*

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & Interface Layer"]
        A1["🌐 Web Landing Page (endoai-landing.html)"]
        A2["💬 Clinical Assistant UI (endoai-assistant.html)"]
        A3["📖 Interactive Swagger Docs (/docs)"]
        A4["⚡ Vercel Edge Global CDN (endoai.vercel.app)"]
    end

    subgraph IngressGateway ["Ingress & Application Gateway"]
        B1["🛡️ Azure Container Apps Ingress (HTTPS / Port 8000)"]
        B2["⚡ FastAPI Application Entrypoint (app/main.py)"]
        B3["⏱️ Instant Startup Lifespan (<0.1s Probe Success)"]
    end

    subgraph DataIngestion ["Offline / Async Ingestion Layer"]
        C1["📄 Clinical PDFs (ESC.pdf & NICE.pdf)"]
        C2["📑 PyMuPDF Text & Structural Extractor"]
        C3["✂️ Semantic Header Chunking (900 tok, 150 overlap)"]
        C4["🧹 Hash & Near-Duplicate Deduplication (841 Chunks)"]
        C5["🧬 Dense Vector Embeddings (NeuML/pubmedbert-base)"]
        C6["🗄️ ChromaDB Vector Store + BM25 Inverted Index"]
    end

    subgraph QueryPipeline ["Online Clinical RAG Pipeline"]
        D1["🔍 User Clinical Query"]
        D2["🔄 Hybrid Search (BM25 Lexical + PubMedBERT Vector)"]
        D3["📊 Top-k Retrieval (k=8 Chunks)"]
        D4["🎯 Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)"]
        D5["🧠 Clinical Evaluator (Azure OpenAI o4-mini)"]
        D6{"Is Evidence Sufficient?"}
        D7["🛑 Refusal Guardrail (Safety Gap & Nearby Sections)"]
        D8["✍️ Grounded Generator (Pydantic Structured Output)"]
        D9["🔬 Citation Verifier (Contiguous Substring Match)"]
    end

    subgraph CloudInfra ["Azure Cloud Infrastructure"]
        E1["📦 Azure Container Registry (acrrag10570)"]
        E2["☁️ Container App: rag-backend-api (depi_demo)"]
        E3["🤖 Azure OpenAI Service (o4-mini Instance)"]
    end

    A1 & A2 & A3 & A4 --> B1
    B1 --> B2
    B2 --> B3
    B3 -.->|Background Thread| C1
    C1 --> C2 --> C3 --> C4 --> C5 --> C6

    B2 --> D1
    D1 --> D2
    C6 --> D2
    D2 --> D3 --> D4 --> D5 --> D6
    D6 -- No --> D7
    D6 -- Yes --> D8 --> D9
    D7 & D9 --> B2

    E1 --> E2
    E2 --> B1
    D5 & D8 <--> E3
```

---

## 2. Core Architectural Pillars

### A. Ingestion & Document Preprocessing
1. **Multi-Guideline Ingestion**:
   - Extracts structured textual content and clinical hierarchies from `ESC.pdf` (European Society of Cardiology 2023 Guidelines, 98 pages) and `NICE.pdf` (National Institute for Health and Care Excellence CG64).
2. **Header-Aware Chunking & Deduplication**:
   - Uses PyMuPDF with font-size heuristic detection to identify clinical section headers (e.g., *Diagnostic Criteria*, *Prophylaxis*, *Urgent Surgery*).
   - Generates chunks of 900 characters with 150-character overlap.
   - Applies MD5 exact deduplication and normalized n-gram near-duplicate removal, resulting in **841 clean guideline passages**.
3. **Dual-Index Creation**:
   - **Dense Index**: Embedded via `NeuML/pubmedbert-base-embeddings` and persisted in **ChromaDB**.
   - **Sparse Index**: Indexed via **BM25Okapi** with tuned parameters ($k_1=1.5, b=0.75$) to capture exact medical terminology, acronyms (IE, PVE, NVE, TEE, TTE), and drug dosages.

---

### B. Hybrid Retrieval & Cross-Encoder Reranking
```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Clinician / User
    participant Router as API Router (/api/v1/ask)
    participant Hybrid as Hybrid Search Engine
    participant Chroma as ChromaDB (Dense)
    participant BM25 as BM25 (Sparse)
    participant Reranker as Cross-Encoder Reranker
    participant LLM as Azure OpenAI (o4-mini)
    participant Verifier as Citation Verifier

    Clinician->>Router: POST /api/v1/ask {"question": "..."}
    Router->>Hybrid: Retrieve Candidates (k=8)
    par Dense Vector Search
        Hybrid->>Chroma: PubMedBERT Embeddings Query
        Chroma-->>Hybrid: Top Dense Passages
    and Lexical Search
        Hybrid->>BM25: Tokenized Keyword Query
        BM25-->>Hybrid: Top Sparse Passages
    end
    Hybrid->>Reranker: Combined Candidates Pool
    Reranker-->>Hybrid: Re-scored & Ordered Passages
    Hybrid->>LLM: Pass Top Reranked Context + Question
    LLM->>LLM: Reason on evidence sufficiency & extract claims
    LLM-->>Router: Structured Clinical Recommendation
    Router->>Verifier: Validate inline citations against raw chunks
    Verifier-->>Router: Verification Log & Exact Excerpts
    Router-->>Clinician: JSON Answer + Confidence + References
```

---

### C. Clinical Reasoning & Safety Guardrail Layer
* **Model**: Azure OpenAI `o4-mini` via Structured Outputs (`client.beta.chat.completions.parse`).
* **Strict Evaluation Protocol**:
  1. The model evaluates whether the retrieved chunks provide explicit, conclusive evidence.
  2. If evidence is ambiguous or missing (e.g., asking for an undocumented medication dose), the system triggers the **Refusal Guardrail**:
     - `refused: true`
     - Returns `evidence_gap` specifying exactly what information is missing.
     - Returns `evidence_found_nearby` suggesting related sections found in the guidelines.
* **Grounded Generation**:
  - Structured output schemas guarantee that every clinical claim has an associated document name, section, and page number.

---

### D. Citation Verifier (Zero Hallucination Guarantee)
* Every citation returned by the generator is checked against the raw chunk text using **Longest Contiguous Substring Matching (LCS)**.
* Matches with a similarity ratio $\ge 0.70$ are verified and linked to page numbers.
* Discrepancies are flagged in the `verification_log`, ensuring zero ungrounded statements reach the clinician.

---

## 3. Infrastructure & Deployment Architecture

```mermaid
graph LR
    subgraph DeveloperWorkspace ["Developer Environment"]
        Dev["Git Commit & Push (GitHub)"]
    end

    subgraph AzureCloud ["Microsoft Azure Cloud (germanywestcentral)"]
        subgraph ACR ["Azure Container Registry"]
            Registry["acrrag10570.azurecr.io/rag-backend:v6"]
        end

        subgraph ContainerApp ["Azure Container Apps (depi_demo)"]
            FastAPIServer["FastAPI + Uvicorn Server (Port 8000)"]
            ChromaEngine["ChromaDB Vector Store"]
            Thread["Background Index Daemon"]
        end

        subgraph AzureAI ["Azure Cognitive Services (swedencentral)"]
            OpenAIInstance["Azure OpenAI o4-mini Deployment"]
        end
    end

    subgraph EdgeLayer ["Global Edge Network"]
        VercelCDN["Vercel Global CDN (endoai.vercel.app)"]
    end

    Dev -->|Docker Build| Registry
    Registry -->|Deploy Image| ContainerApp
    FastAPIServer <-->|HTTPS REST API| OpenAIInstance
    FastAPIServer --> ChromaEngine
    VercelCDN <-->|CORS REST API| FastAPIServer
```

---

## 4. API Endpoints Reference

| HTTP Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive Web Landing Page |
| `GET` | `/assistant` | Serves the Clinical AI Assistant chat interface |
| `GET` | `/api/v1/health` | Health probe reporting server status, index readiness, and chunk count |
| `POST` | `/api/v1/ask` | Clinical RAG QA pipeline with hybrid retrieval & structured response |
| `GET` | `/api/v1/documents` | Lists all indexed medical guideline PDF documents |
| `GET` | `/api/v1/documents/{filename}` | Streams raw guideline PDF for in-browser inspection |
| `GET` | `/api/v1/collections/stats` | Vector store metrics, embedding model, and persist directory stats |
| `POST` | `/api/v1/reindex` | Triggers background index re-extraction and embedding |
| `GET` | `/docs` | Interactive OpenAPI 3.1 Swagger UI documentation |

---

## 5. Non-Blocking Startup Mechanism

To prevent Cloud Gateway timeouts (e.g., Azure 504 Ingress Timeout), the server separates networking from model loading:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Server listens on HTTP port immediately in <0.1s
    # 2. Azure health probes succeed on first attempt
    t = threading.Thread(target=_build_rag_background, args=(app,), daemon=True)
    t.start()
    yield
```

* While the index builds in the background thread, `/api/v1/health` returns `index_ready: false`.
* Once completed, `/api/v1/health` flips to `index_ready: true (841 chunks)`, enabling full query processing.

---

## 6. Technology Stack Summary

| Layer | Technology |
| :--- | :--- |
| **Language & Framework** | Python 3.11, FastAPI, Uvicorn, Pydantic v2 |
| **PDF Extraction** | PyMuPDF (fitz) |
| **Embeddings** | `NeuML/pubmedbert-base-embeddings` (HuggingFace Transformers / PyTorch) |
| **Vector Database** | ChromaDB (Persistent Disk Storage) |
| **Lexical Search** | Rank-BM25 (BM25Okapi) |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM Reasoning** | Azure OpenAI `o4-mini` (Structured Outputs & Function Calling) |
| **Containerization** | Multi-stage Docker (Python 3.11 Slim + libgomp1) |
| **Cloud Hosting** | Azure Container Apps, Azure Container Registry, Azure OpenAI |
| **Frontend UI** | HTML5, Vanilla CSS3 (Custom Design System, Glassmorphism, Responsive Drawer), JavaScript ES6 |
| **Edge Distribution** | Vercel Global Edge Network |
