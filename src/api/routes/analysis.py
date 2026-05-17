import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agents.graph import run_pipeline
from src.rag.retriever import HybridRetriever

router = APIRouter()


class AnalyzeRequest(BaseModel):
    document_id: str
    content: str
    company_ticker: str = ""


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    modality_filter: str | None = None


@router.post("/analyze")
async def analyze_document(request: AnalyzeRequest):
    try:
        result = run_pipeline(
            document_id=request.document_id,
            content=request.content,
            company_ticker=request.company_ticker,
        )
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query_knowledge_base(request: QueryRequest):
    start = time.time()
    retriever = HybridRetriever()
    results = retriever.retrieve(
        query=request.query,
        modality_filter=request.modality_filter,
    )
    elapsed = time.time() - start
    return {
        "query": request.query,
        "results": results[: request.top_k],
        "latency_seconds": round(elapsed, 3),
    }
