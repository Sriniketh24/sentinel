from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from src.ingestion.tasks import ingest_filing

router = APIRouter()


class IngestRequest(BaseModel):
    ticker: str
    form_types: list[str] = ["10-K", "10-Q"]
    limit: int = 5


class IngestResponse(BaseModel):
    task_id: str
    status: str


@router.post("/ingest/edgar", response_model=IngestResponse)
async def ingest_from_edgar(request: IngestRequest):
    task = ingest_filing.delay(
        ticker=request.ticker,
        form_types=request.form_types,
        limit=request.limit,
    )
    return IngestResponse(task_id=task.id, status="queued")


@router.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)):
    import uuid
    from pathlib import Path

    upload_dir = Path("data/raw/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    ext = Path(file.filename or "doc").suffix
    dest = upload_dir / f"{file_id}{ext}"
    content = await file.read()
    dest.write_bytes(content)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "path": str(dest),
    }


@router.get("/ingest/status/{task_id}")
async def ingest_status(task_id: str):
    from celery.result import AsyncResult

    result = AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }
