from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "sentinel"}


@router.get("/ready")
async def readiness_check():
    return {"status": "ready"}
