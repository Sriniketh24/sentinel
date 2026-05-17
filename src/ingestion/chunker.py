import uuid

import structlog

from src.config import get_settings
from src.models.document import DocumentChunk, Modality

log = structlog.get_logger()


class MultimodalChunker:
    def __init__(self) -> None:
        settings = get_settings()
        self._chunk_size = settings.chunk_size
        self._overlap = settings.chunk_overlap

    def chunk(self, raw_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        result: list[DocumentChunk] = []
        for raw in raw_chunks:
            if raw.modality == Modality.TEXT:
                result.extend(self._split_text(raw))
            else:
                raw.id = raw.id or str(uuid.uuid4())
                result.append(raw)

        for i, c in enumerate(result):
            c.chunk_index = i
            if not c.id:
                c.id = str(uuid.uuid4())

        log.info("chunked documents", input=len(raw_chunks), output=len(result))
        return result

    def _split_text(self, chunk: DocumentChunk) -> list[DocumentChunk]:
        text = chunk.content
        words = text.split()

        if len(words) <= self._chunk_size:
            return [chunk]

        pieces: list[DocumentChunk] = []
        start = 0
        while start < len(words):
            end = min(start + self._chunk_size, len(words))
            segment = " ".join(words[start:end])
            pieces.append(
                DocumentChunk(
                    id=f"{chunk.document_id}_c{len(pieces)}",
                    document_id=chunk.document_id,
                    content=segment,
                    modality=chunk.modality,
                    page_number=chunk.page_number,
                    metadata={**chunk.metadata, "chunk_part": len(pieces)},
                )
            )
            start = end - self._overlap if end < len(words) else end

        return pieces
