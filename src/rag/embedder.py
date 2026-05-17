import structlog
from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.models.document import DocumentChunk

log = structlog.get_logger()


class Embedder:
    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.embedding_model
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("loading embedding model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    def embed_text(self, text: str) -> list[float]:
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        texts = [c.content for c in chunks]
        embeddings = self.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        log.info("embedded chunks", count=len(chunks))
        return chunks
