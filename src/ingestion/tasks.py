import asyncio

from celery import Celery

from src.config import get_settings

settings = get_settings()
celery_app = Celery("sentinel", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"


@celery_app.task(name="ingest_filing", bind=True, max_retries=3)
def ingest_filing(self, ticker: str, form_types: list[str] | None = None, limit: int = 5):
    from src.ingestion.edgar import EdgarClient
    from src.ingestion.parsers.pdf_parser import PDFParser
    from src.ingestion.chunker import MultimodalChunker
    from src.rag.embedder import Embedder
    from src.rag.store import VectorStore

    async def _run():
        client = EdgarClient()
        documents = await client.fetch_filings(ticker, form_types, limit)

        pdf_parser = PDFParser()
        chunker = MultimodalChunker()
        embedder = Embedder()
        store = VectorStore()

        total_chunks = 0
        for doc in documents:
            if doc.file_path:
                raw_chunks = pdf_parser.parse(doc.file_path, doc.id)
                chunks = chunker.chunk(raw_chunks)
                embedded = embedder.embed_chunks(chunks)
                await store.upsert(embedded)
                total_chunks += len(embedded)

        return {"ticker": ticker, "documents": len(documents), "chunks": total_chunks}

    return asyncio.get_event_loop().run_until_complete(_run())
