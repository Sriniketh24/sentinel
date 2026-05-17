# Sentinel: Multimodal Financial Document Intelligence Pipeline

An end-to-end AI system that ingests SEC filings, processes them through a multi-agent orchestration pipeline, and produces automated equity research briefs with cited evidence chains and confidence scores.

## Key Features

- **Multi-Agent Orchestration** (LangGraph): Classify > Extract > Verify > Synthesize pipeline with conditional routing
- **Multimodal RAG**: BGE embeddings + Qdrant vector store + cross-encoder reranking (MRR@5 = 1.0)
- **Real SEC Data**: Live ingestion from EDGAR API (10-K, 10-Q, 8-K filings)
- **LLM Synthesis**: Zephyr-7B via HF Inference API with template fallback
- **Production Patterns**: Async task queue, structured logging, Prometheus metrics, Docker deployment
- **Eval Harness**: Golden dataset scoring with MRR, NDCG, entity F1, latency percentiles

## Architecture

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│  SEC EDGAR API │    │  File Upload   │    │  Audio (ASR)   │
└───────┬────────┘    └───────┬────────┘    └───────┬────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Ingestion & Chunking Layer                       │
│  HTML Parser │ PDF Parser │ Word Splitter (512w, 64 overlap) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           Embedding & Vector Store (Qdrant)                   │
│  BGE-small-en-v1.5 (384-dim) │ Cross-encoder reranking       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         Multi-Agent Pipeline (LangGraph StateGraph)           │
│                                                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │Classifier│ → │Extractor │ → │ Verifier │ → │Synthesizer│ │
│  │(zero-shot│   │(NER +    │   │(RAG cross│   │(LLM brief │ │
���  │ classify)│   │ regex)   │   │ -ref)    │   │ + fallback│ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI (7 endpoints) │ Prometheus │ Celery + Redis          │
└─────────────────────────────────────────────────────────────┘
```

## Hugging Face Models Used

| Model | Task | Role |
|-------|------|------|
| `typeform/distilbert-base-uncased-mnli` | Zero-Shot Classification | Document type routing + sentiment detection |
| `dslim/bert-base-NER` | Token Classification (NER) | Entity extraction (companies, metrics, dates) |
| `BAAI/bge-small-en-v1.5` | Feature Extraction | Document embeddings for RAG (384-dim) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Text Ranking | Rerank retrieved passages |
| `HuggingFaceH4/zephyr-7b-beta` | Text Generation | Research brief synthesis |

## Quick Start

```bash
# Clone and install
git clone https://github.com/sriniketh01/sentinel.git
cd sentinel
pip install -e ".[dev]"

# Start infrastructure
docker compose -f docker/docker-compose.yml up -d

# Ingest real SEC filings
python scripts/ingest_live.py

# Run the pipeline on a document
python scripts/run_pipeline.py

# Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Run eval harness
python scripts/run_eval.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/ready` | Readiness probe |
| POST | `/api/v1/analyze` | Full pipeline analysis |
| POST | `/api/v1/query` | RAG knowledge base query |
| POST | `/api/v1/ingest/edgar` | Queue SEC filing ingestion |
| POST | `/api/v1/ingest/upload` | Upload document for processing |
| GET | `/metrics/` | Prometheus metrics |

### Example: Analyze a Document

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "aapl-10q-2026",
    "content": "Apple Inc reported revenue of $95,359 million...",
    "company_ticker": "AAPL"
  }'
```

### Example: Query the Knowledge Base

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple total revenue?", "top_k": 3}'
```

## Eval Metrics

Evaluated against a golden dataset of 5 financial queries over 6 real SEC filings (658 vectors):

| Metric | Score |
|--------|-------|
| Retrieval MRR@5 | 1.000 |
| Retrieval NDCG@10 | 1.000 |
| Extraction F1 (Entity) | 0.613 |
| Latency p50 | 32.9s (CPU) |
| Latency p95 | 42.3s (CPU) |

## Production Deployment

```bash
# Full production stack with health checks and resource limits
docker compose -f docker/docker-compose.prod.yml up -d --build
```

## Project Structure

```
sentinel/
├── src/
│   ├── agents/          # LangGraph multi-agent pipeline
│   │   ├── classifier.py    # Zero-shot document/sentiment classification
│   │   ├── extractor.py     # NER + regex financial extraction
│   │   ├── verifier.py      # RAG-based claim verification
│   │   ├── synthesizer.py   # LLM brief generation
│   │   ├── graph.py         # LangGraph StateGraph orchestration
│   │   └── state.py         # Pipeline state model
│   ├── api/             # FastAPI application
│   │   ├── main.py
│   │   └── routes/
│   ├── config/          # Settings and logging
│   ├── eval/            # Evaluation harness
│   │   ├── runner.py        # Eval orchestration
│   │   ├── scorer.py        # MRR, NDCG, F1 scorers
│   │   └── metrics.py       # Metrics dataclass
│   ├── ingestion/       # Document ingestion
│   │   ├── edgar.py         # SEC EDGAR client
│   │   ├── chunker.py       # Text chunking
│   │   ├── tasks.py         # Celery tasks
│   │   └── parsers/
│   ├── models/          # Pydantic data models
│   └── rag/             # RAG components
│       ├── embedder.py      # BGE embedding
│       ├── store.py         # Qdrant vector store
│       └── retriever.py     # Hybrid retrieval + reranking
├── scripts/             # CLI scripts
├── tests/               # Unit tests (16 passing)
├── docker/              # Docker configs
├── data/eval/           # Golden evaluation dataset
├── Dockerfile           # Production container
└── pyproject.toml       # Dependencies and config
```

## Tech Stack

- **Orchestration**: LangGraph (stateful multi-agent graphs)
- **Vector Store**: Qdrant (cosine similarity, metadata filtering)
- **Embeddings**: sentence-transformers (BGE-small-en-v1.5)
- **Reranking**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM**: HuggingFace Inference API (Zephyr-7B)
- **NLP**: Transformers (DistilBERT-MNLI, BERT-NER)
- **API**: FastAPI + Uvicorn
- **Queue**: Celery + Redis
- **Observability**: Prometheus + structlog
- **Testing**: pytest + coverage
- **CI/CD**: GitHub Actions

## License

MIT
