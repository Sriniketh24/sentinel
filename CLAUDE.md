# Sentinel: Multimodal Financial Document Intelligence Pipeline

## What It Does

An end-to-end system that ingests mixed-format financial documents — SEC filings (PDFs with tables/charts), earnings call audio, analyst slide decks, and news articles — then uses multi-agent orchestration to extract structured insights, cross-reference claims via multimodal RAG, and produce automated equity research briefs with cited evidence chains and confidence scores.

**Real-world problem**: Analysts at investment firms spend 60-80% of their time manually reading filings, listening to earnings calls, and cross-referencing data across documents. This system automates the extraction-and-synthesis loop while keeping humans in the decision seat.

## Hugging Face Tasks Used

| Task | Role in Pipeline |
|------|-----------------|
| Automatic Speech Recognition | Transcribe + diarize earnings call audio (speaker-attributed Q&A) |
| Document Question Answering | Extract answers from PDF filings (10-K, 10-Q, 8-K) |
| Table Question Answering | Query financial tables embedded in filings (revenue breakdowns, balance sheets) |
| Image-to-Text / Visual QA | Interpret charts, graphs, and infographics in slide decks and reports |
| Text Classification | Route documents by type, detect sentiment, flag risk language |
| Zero-Shot Classification | Categorize novel document types without retraining |
| Summarization | Condense long filings into digestible sections |
| Feature Extraction | Generate embeddings for text chunks (RAG vector store) |
| Image Feature Extraction | Generate embeddings for chart/figure retrieval |
| Sentence Similarity + Text Ranking | Hybrid retrieval with cross-encoder reranking |
| Text Generation | Produce final research briefs and natural-language answers |
| Token Classification | NER for company names, ticker symbols, financial metrics, dates |

## Architecture

```
                        ┌─────────────────────────────┐
                        │     Ingestion Gateway        │
                        │  (FastAPI + Celery workers)  │
                        └──────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
     ┌──────────────┐   ┌──────────────┐     ┌──────────────┐
     │  PDF Parser   │   │  Audio ASR   │     │  Image/Chart │
     │  + Table QA   │   │  + Diarize   │     │  Interpreter │
     │  (Unstructured│   │  (Whisper v4) │     │  (Florence-3)│
     │   + LayoutLM) │   └──────┬───────┘     └──────┬───────┘
     └──────┬───────┘          │                     │
            │                  │                     │
            ▼                  ▼                     ▼
     ┌─────────────────────────────────────────────────────┐
     │            Multimodal Chunking & Embedding           │
     │   (text chunks + table snapshots + chart captions)   │
     │        → pgvector / Qdrant hybrid index              │
     └──────────────────────┬──────────────────────────────┘
                            │
                            ▼
     ┌─────────────────────────────────────────────────────┐
     │           Multi-Agent Orchestration (LangGraph)      │
     │                                                      │
     │  ┌───────────┐  ┌───────────┐  ┌──────────────┐    │
     │  │ Classifier │→ │ Extractor │→ │  Verifier    │    │
     │  │   Agent    │  │   Agent   │  │   Agent      │    │
     │  │(route docs,│  │(pull KPIs,│  │(cross-ref    │    │
     │  │ detect     │  │ claims,   │  │ against RAG  │    │
     │  │ sentiment) │  │ guidance) │  │ + flag gaps)  │    │
     │  └───────────┘  └───────────┘  └──────┬───────┘    │
     │                                        │            │
     │                              ┌─────────▼─────────┐  │
     │                              │  Synthesis Agent   │  │
     │                              │  (generate brief   │  │
     │                              │   with citations)  │  │
     │                              └───────────────────┘  │
     └─────────────────────────────────────────────────────┘
                            │
                            ▼
     ┌─────────────────────────────────────────────────────┐
     │              Eval & Observability Layer               │
     │  LangSmith traces │ custom scorers │ Prometheus      │
     │  extraction acc.   │ retrieval MRR  │ cost/token dash │
     └─────────────────────────────────────────────────────┘
```

## Recommended Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Orchestration | LangGraph (stateful agents) | Graph-based agent coordination with checkpointing, retries, human-in-the-loop |
| RAG / Retrieval | LlamaIndex (multimodal indexing) + Qdrant | Native support for text + image embeddings, metadata filtering, hybrid search |
| Document Parsing | Unstructured.io + pdfplumber | Production-grade PDF/table extraction with layout awareness |
| Vector Store | Qdrant (or pgvector for simplicity) | Multimodal vector support, filtering, quantization |
| Models | HF Inference API + local Transformers | Whisper v3-large (ASR), LayoutLMv3 (DocQA), Florence-2 (VQA), Mistral/Llama (generation), BGE-M3 (embeddings) |
| Eval Harness | RAGAS + custom scorers + LangSmith | Retrieval quality, faithfulness, answer relevance, extraction accuracy |
| API | FastAPI + Celery + Redis | Async document processing, task queuing, rate limiting |
| Observability | LangSmith + Prometheus + Grafana | Trace every agent step, track cost/latency/quality per query |
| Data Sources | SEC EDGAR API, Financial Modeling Prep API, Seeking Alpha RSS | Free, real-world financial data |
| Infra | Docker Compose → optional K8s | Reproducible local dev, scales to cloud |

## What Makes This Stand Out

1. **Genuinely multimodal, not just text-in-text-out.** You process PDFs (layout-aware), audio (diarized transcripts), and chart images (vision models) into a unified knowledge store.
2. **Multi-agent with clear justification.** Each agent has a distinct capability and failure mode.
3. **Evaluation is a first-class feature, not an afterthought.** Golden dataset of ~200 annotated filing questions, extraction accuracy, retrieval MRR/NDCG, faithfulness scores, cost-per-query.
4. **Production patterns throughout.** Async ingestion queue, retry logic, structured logging, cost tracking, model versioning, A/B testing.
5. **Real data, real complexity.** SEC filings have messy tables, footnotes, cross-references, and legalese.

## Resume-Ready Impact Metrics

- Built multimodal RAG pipeline processing 4 document modalities (PDF, audio, image, tabular) with 87%+ extraction accuracy on a 200-question golden eval set
- Designed multi-agent orchestration layer (LangGraph) reducing manual document analysis time by ~70% across 500+ SEC filings
- Achieved retrieval MRR@10 of 0.82 with hybrid search (dense + sparse + cross-encoder reranking), 18% improvement over baseline dense retrieval
- Implemented cost-aware model routing cutting inference cost by 45% while maintaining quality
- Built end-to-end eval harness tracking faithfulness, relevance, and extraction accuracy with automated regression detection
- Processed 10,000+ pages of financial documents with p95 latency under 8 seconds per query

## 3-Month Timeline

| Weeks | Milestone |
|-------|-----------|
| 1-2 | Data pipeline: SEC EDGAR ingestion, PDF/table parsing, basic chunking → vector store |
| 3-4 | Audio pipeline: ASR integration, diarization, transcript chunking |
| 5-6 | Multimodal RAG: Hybrid retrieval, chart/image embedding, reranking |
| 7-8 | Multi-agent orchestration: LangGraph agents, state management, error handling |
| 9-10 | Eval harness: Golden dataset creation, RAGAS integration, custom scorers |
| 11-12 | Production polish: Cost tracking, caching, Docker deployment, Grafana dashboard, README + demo video |
