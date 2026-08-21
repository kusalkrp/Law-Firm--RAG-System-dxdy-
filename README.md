# Law Firm RAG System

An enterprise-grade, hallucination-resistant **Retrieval-Augmented Generation (RAG)** pipeline designed for legal contract analysis and risk assessment. 

Built with **FastAPI**, **ChromaDB**, **BM25**, **Cohere Rerank**, and **Claude Sonnet 4.6** via the **Vercel AI Gateway**.

---

## 🏛️ Architecture Overview

```
                               ┌─────────────────────────────┐
                               │        FastAPI Server       │
                               │  POST /query  POST /ingest  │
                               │         GET /health         │
                               └──────────────┬──────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         │                                    │                                    │
 ┌───────▼────────┐                 ┌─────────▼─────────┐                ┌─────────▼─────────┐
 │ Document       │                 │ Hybrid Retriever  │                │ Response Generator│
 │ Processor      │                 │                   │                │                   │
 │ - PDF parsing  │                 │ Dense (ChromaDB)─┐│                │ Grounded prompt   │
 │ - Clause chunk │                 │ Sparse (BM25)────┼┼─▶ RRF Fusion   │ Citation contract │
 │ - Metadata tag │                 │                  ││   ──▶ Rerank   │ CRAG pre-check    │
 └───────┬────────┘                 └─────────┬─────────┘  (Cohere v3.5) └─────────┬─────────┘
         │                                    │                                    │
         │                                    │                          ┌─────────▼─────────┐
         │                                    │                          │ Faithfulness      │
         │                                    │                          │ Evaluator         │
         │                                    │                          │ (Claim Decomp)    │
         ▼                                    ▼                          └───────────────────┘
 ┌───────────────────┐              ┌───────────────────┐
 │  ChromaDB Server  │              │    BM25 Index     │
 │  (Vector Store,   │              │ (Keyword, in-app) │
 │   Docker Service) │              └───────────────────┘
 └───────────────────┘
```

### Key Engineering Features
1. **Hybrid Retrieval (Dense + Sparse)**: Combines semantic vector similarity (`openai/text-embedding-3-large`) with exact keyword matching (`BM25Okapi`) merged via **Reciprocal Rank Fusion (RRF, $k=60$)**.
2. **Cross-Encoder Reranking**: Re-scores the top-20 fused candidates using `cohere/rerank-v3.5` via the Gateway to guarantee high precision.
3. **Structure-Aware Legal Chunking**: Preserves section and clause boundaries with metadata enrichment (`source_file`, `page_number`, `section_heading`, `clause_type`).
4. **Hallucination Prevention**:
   - **CRAG Relevance Pre-Check**: Short-circuits irrelevant queries straight to abstention before calling the generation LLM.
   - **Strict Citation Contract**: System prompt mandates verbatim citations with specific document and page numbers.
   - **Post-Hoc Verification**: Prunes citations that do not verifiably appear in the retrieved context passages.
   - **Deterministic Decoding**: Zero-temperature generation (`temperature=0.0`).
5. **Claim-Level Faithfulness Evaluation**: Decomposes answers into atomic claims and verifies each via LLM-as-judge ($0.0 \dots 1.0$).
6. **Containerized Deployment**: Multi-stage Dockerfile and Docker Compose orchestration separating the application and ChromaDB services.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.12+**
- **Docker & Docker Compose**
- **Vercel AI Gateway API Key**

### 🖥️ Interactive Web UI Dashboard
The system includes a modern, rich web interface at:
- **Web UI**: [http://localhost:8001/](http://localhost:8001/) or [http://localhost:8001/ui](http://localhost:8001/ui)
- **Interactive API Docs (Swagger)**: [http://localhost:8001/docs](http://localhost:8001/docs)

Features available in the UI:
- **Curated Legal Presets**: 1-click execution of benchmark queries (Termination, Indemnity, Severance, Confidentiality, Out-of-scope Abstention).
- **Interactive Citation Cards**: Visual display of source PDF, page number, clause type, and exact excerpt.
- **Real-Time Guardrail Metrics**: Faithfulness score gauge, confidence badge, latency counter, and abstention alerts.
- **Contract Re-indexing**: Instant 1-click document ingestion into ChromaDB and BM25.

### 2. Environment Configuration
Copy `.env.example` to `.env` and enter your Vercel AI Gateway key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
AI_GATEWAY_API_KEY=your_vercel_ai_gateway_key_here
AI_GATEWAY_BASE_URL=https://ai-gateway.vercel.sh/v1

LLM_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=openai/text-embedding-3-small
RERANK_MODEL=cohere/rerank-v3.5

CHROMA_HOST=localhost
CHROMA_PORT=8000
```

---

## 🐳 Running with Docker Compose (Recommended)

Start both the dedicated ChromaDB server and the FastAPI application:
```bash
docker compose up --build -d
```
The API will be available at `http://localhost:8001` (with interactive Swagger docs at `http://localhost:8001/docs`).

Check health:
```bash
curl http://localhost:8001/health
```

---

## 💻 Running Locally (Development)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start ChromaDB container**:
   ```bash
   docker run -d --name chromadb -p 8000:8000 chromadb/chroma
   ```

3. **Start FastAPI application**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

---

## 📑 API Reference

### 1. Ingest Contracts — `POST /ingest`
Parses PDF contracts from `data/contracts/`, creates dense embeddings, stores them in ChromaDB, and builds the BM25 index.

```bash
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "data/contracts"}'
```

**Response**:
```json
{
  "documents_processed": 5,
  "chunks_created": 78,
  "status": "success",
  "message": "Ingested 5 documents (78 chunks)."
}
```

---

### 2. Query Contracts — `POST /query`
Performs hybrid retrieval, generates a cited answer, and computes live faithfulness score.

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the indemnification obligations in the Manufacturing Agreement?"
  }'
```

**Response**:
```json
{
  "question": "What are the indemnification obligations in the Manufacturing Agreement?",
  "answer": "Under the Manufacturing Agreement, Premier and Heritage have reciprocal indemnification obligations. Premier must indemnify, defend, and hold harmless Heritage from any claims arising out of Premier's breach or negligence [BellringBrandsInc, p.7]. Heritage holds corresponding indemnification duties for Premier [BellringBrandsInc, p.7].",
  "citations": [
    {
      "source_file": "BellringBrandsInc_20190920_S-1_EX-10.12_11817081_EX-10.12_Manufacturing Agreement1.pdf",
      "page": 7,
      "section": "Section 12. Indemnification",
      "clause_type": "indemnification",
      "text_excerpt": "Premier shall indemnify, defend and hold Heritage harmless from any claims..."
    }
  ],
  "faithfulness_score": 1.0,
  "confidence": "high",
  "abstention_triggered_by": null,
  "processing_time_ms": 1420.5
}
```

---

### 3. Out-of-Scope Query (Abstention Handling)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

**Response**:
```json
{
  "question": "What is the capital of France?",
  "answer": "I cannot find this information in the provided documents.",
  "citations": [],
  "faithfulness_score": 1.0,
  "confidence": "low",
  "abstention_triggered_by": "pre_check",
  "processing_time_ms": 350.2
}
```

---

## 🧪 Testing & Evaluation

### Run Unit and Integration Tests
```bash
pytest tests/ -v
```

### Run Methodical Benchmark Evaluation
Runs the 15-question golden QA dataset covering all contracts and computes aggregate faithfulness and citation precision:
```bash
python eval/run_ragas.py
```

### Run Retrieval Spot Check
```bash
python eval/run_retrieval_check.py
```

---

## 📁 Repository Structure

```
.
├── app/
│   ├── api/
│   │   ├── routes.py            # API route definitions (/health, /query, /ingest)
│   │   └── schemas.py           # Pydantic request/response models
│   ├── core/
│   │   ├── document_processor.py# PDF parsing, structure-aware chunking, clause tagging
│   │   ├── embeddings.py        # Gateway embedding client wrapper
│   │   ├── evaluator.py         # Claim-level faithfulness evaluator
│   │   ├── generator.py         # Grounded prompt generation & citation verification
│   │   └── retriever.py         # Hybrid retriever (Dense + BM25 + RRF + Cohere rerank)
│   ├── storage/
│   │   ├── bm25_index.py        # In-memory + persistent BM25 keyword index
│   │   └── vector_store.py      # ChromaDB HTTP client wrapper
│   ├── config.py                # Typed settings & startup model validator
│   └── main.py                  # FastAPI entry point & lifespan
├── data/
│   └── contracts/               # PDF legal contract files
├── eval/
│   ├── golden_qa.json           # 15-question golden evaluation dataset
│   ├── retrieval_spot_check.json# Early retrieval quality test cases
│   ├── run_ragas.py             # Methodical evaluation runner
│   └── run_retrieval_check.py   # Retrieval Hit@5 verification runner
├── tests/
│   ├── test_pipeline.py         # End-to-end query & abstention tests
│   ├── test_retriever.py        # Hybrid retrieval tests
│   └── test_storage.py          # Vector store & BM25 unit tests
├── Dockerfile                   # Multi-stage container definition
├── docker-compose.yml           # App + ChromaDB multi-container orchestration
├── requirements.txt             # Pinned python dependencies
├── README.md                    # Setup & user documentation
└── REPORT.md                    # Architecture & evaluation approach report
```