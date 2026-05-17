web: uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
worker: celery -A src.ingestion.tasks worker --loglevel=info --concurrency=2
