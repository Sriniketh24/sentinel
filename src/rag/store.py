import structlog
from qdrant_client import QdrantClient, models

from src.config import get_settings
from src.models.document import DocumentChunk

log = structlog.get_logger()

VECTOR_SIZE = 384  # bge-small-en-v1.5 dimension


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._qdrant_url = settings.qdrant_url
        self._collection = settings.qdrant_collection
        self._client: QdrantClient | None = None
        self._available: bool | None = None

    def _get_client(self) -> QdrantClient | None:
        if self._available is False:
            return None
        if self._client is None:
            try:
                self._client = QdrantClient(
                    url=self._qdrant_url, check_compatibility=False, timeout=5
                )
                self._ensure_collection()
                self._available = True
            except Exception as e:
                log.warning("Qdrant unavailable", error=str(e)[:120])
                self._available = False
                return None
        return self._client

    def _ensure_collection(self) -> None:
        if self._client is None:
            return
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )
            log.info("created collection", name=self._collection)

    async def upsert(self, chunks: list[DocumentChunk]) -> None:
        client = self._get_client()
        if client is None:
            log.warning("Qdrant unavailable, skipping upsert")
            return

        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            points.append(
                models.PointStruct(
                    id=hash(chunk.id) % (2**63),
                    vector=chunk.embedding,
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "modality": chunk.modality.value,
                        "page_number": chunk.page_number,
                        "metadata": chunk.metadata,
                    },
                )
            )

        if points:
            client.upsert(collection_name=self._collection, points=points)
            log.info("upserted vectors", count=len(points))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        modality_filter: str | None = None,
    ) -> list[dict]:
        client = self._get_client()
        if client is None:
            log.info("Qdrant unavailable, returning empty results")
            return []

        query_filter = None
        if modality_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="modality",
                        match=models.MatchValue(value=modality_filter),
                    )
                ]
            )

        try:
            results = client.query_points(
                collection_name=self._collection,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
            )
        except Exception as e:
            log.warning("Qdrant search failed", error=str(e)[:120])
            return []

        return [
            {
                "score": point.score,
                "content": point.payload.get("content", ""),
                "document_id": point.payload.get("document_id", ""),
                "modality": point.payload.get("modality", ""),
                "page_number": point.payload.get("page_number"),
                "metadata": point.payload.get("metadata", {}),
            }
            for point in results.points
        ]
