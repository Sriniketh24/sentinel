from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.api.routes import analysis, health, ingestion
from src.config.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


app = FastAPI(
    title="Sentinel",
    description="Multimodal Financial Document Intelligence Pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(ingestion.router, prefix="/api/v1", tags=["ingestion"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
