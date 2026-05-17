"""Ingest real SEC filings directly (no Celery required)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.logging import setup_logging
from src.ingestion.edgar import EdgarClient
from src.ingestion.parsers.pdf_parser import PDFParser
from src.ingestion.parsers.html_parser import HTMLParser
from src.ingestion.chunker import MultimodalChunker
from src.rag.embedder import Embedder
from src.rag.store import VectorStore

import structlog

log = structlog.get_logger()

TICKERS = ["AAPL", "MSFT", "NVDA"]
FORM_TYPES = ["10-K", "10-Q"]
LIMIT_PER_TICKER = 2


def parse_file(file_path: str, document_id: str, pdf_parser: PDFParser, html_parser: HTMLParser):
    ext = Path(file_path).suffix.lower()
    if ext in (".htm", ".html", ".txt"):
        return html_parser.parse(file_path, document_id)
    elif ext == ".pdf":
        return pdf_parser.parse(file_path, document_id)
    else:
        log.warning("unknown file type, trying html", ext=ext)
        return html_parser.parse(file_path, document_id)


async def main():
    setup_logging()
    log.info("starting live ingestion", tickers=TICKERS)

    client = EdgarClient()
    pdf_parser = PDFParser()
    html_parser = HTMLParser()
    chunker = MultimodalChunker()
    embedder = Embedder()
    store = VectorStore()

    total_docs = 0
    total_chunks = 0

    for ticker in TICKERS:
        log.info("fetching filings", ticker=ticker)
        documents = await client.fetch_filings(ticker, FORM_TYPES, LIMIT_PER_TICKER)
        log.info("fetched", ticker=ticker, count=len(documents))

        for doc in documents:
            if not doc.file_path:
                log.warning("no file path, skipping", doc_id=doc.id)
                continue

            raw_chunks = parse_file(doc.file_path, doc.id, pdf_parser, html_parser)
            if not raw_chunks:
                log.warning("no chunks from parsing", doc_id=doc.id, path=doc.file_path)
                continue

            chunks = chunker.chunk(raw_chunks)
            log.info("chunked", doc_id=doc.id, raw=len(raw_chunks), final=len(chunks))

            embedded = embedder.embed_chunks(chunks)
            await store.upsert(embedded)
            total_chunks += len(embedded)
            total_docs += 1

            log.info(
                "ingested document",
                ticker=ticker,
                doc_type=doc.doc_type.value,
                chunks=len(embedded),
            )

    log.info("ingestion complete", documents=total_docs, total_chunks=total_chunks)


if __name__ == "__main__":
    asyncio.run(main())
