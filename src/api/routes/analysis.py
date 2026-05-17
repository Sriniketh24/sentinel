import time

import structlog
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from src.agents.graph import run_pipeline
from src.parsers.pdf import extract_pdf_text
from src.rag.retriever import HybridRetriever

log = structlog.get_logger()

router = APIRouter()

MAX_PDF_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


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


@router.post("/extract-pdf")
async def extract_pdf(file: UploadFile):
    """Extract text from an uploaded PDF file.

    Returns extracted text with page markers so the user can
    review/edit before sending to /analyze.
    """
    # Validate content type and filename
    is_pdf = (
        (file.content_type and file.content_type == "application/pdf")
        or (file.filename and file.filename.lower().endswith(".pdf"))
    )
    if not is_pdf:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    # Read file with size enforcement (chunk-based)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB chunks
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_PDF_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"PDF exceeds the 50 MB upload limit "
                    f"({total / (1024 * 1024):.1f} MB received so far). "
                    f"Please upload a smaller file."
                ),
            )
        chunks.append(chunk)

    file_bytes = b"".join(chunks)
    log.info("pdf_upload_received", filename=file.filename, size_mb=round(total / (1024 * 1024), 2))

    # Extract text
    try:
        result = extract_pdf_text(file_bytes, filename=file.filename)
    except Exception as e:
        log.error("pdf_extraction_error", error=str(e)[:200])
        raise HTTPException(
            status_code=422,
            detail=f"Failed to extract text from PDF: {str(e)[:200]}",
        )

    # If extraction returned an error (scanned PDF, corrupt file, etc.)
    if not result.get("has_text") and result.get("error"):
        return result  # Returns 200 with has_text=false and error message

    return result


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
