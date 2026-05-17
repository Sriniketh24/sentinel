import math

import structlog
from huggingface_hub import InferenceClient

from src.config import get_settings
from src.models.document import DocumentChunk

log = structlog.get_logger()


class Embedder:
    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.embedding_model
        self._hf_token = settings.hf_api_token
        self._local_model = None

    def embed_text(self, text: str) -> list[float]:
        result = self._embed_via_api(text)
        if result is not None:
            return result
        try:
            return self._embed_local(text)
        except (ImportError, Exception) as e:
            log.warning("All embedding methods failed", error=str(e)[:120])
            return self._zero_vector()

    def embed_texts(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        results = self._embed_texts_via_api(texts)
        if results is not None:
            return results
        try:
            return self._embed_texts_local(texts, batch_size)
        except (ImportError, Exception) as e:
            log.warning("All batch embedding methods failed", error=str(e)[:120])
            return [self._zero_vector() for _ in texts]

    @staticmethod
    def _zero_vector() -> list[float]:
        """Return a zero vector when no embedding method is available."""
        from src.rag.store import VECTOR_SIZE

        return [0.0] * VECTOR_SIZE

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        texts = [c.content for c in chunks]
        embeddings = self.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        log.info("embedded chunks", count=len(chunks))
        return chunks

    def _embed_via_api(self, text: str) -> list[float] | None:
        if not self._hf_token:
            return None
        try:
            client = InferenceClient(token=self._hf_token)
            result = client.feature_extraction(text, model=self._model_name)
            embedding = result[0] if isinstance(result[0], list) else result
            return self._normalize(embedding)
        except Exception as e:
            log.warning("API embedding failed, trying local", error=str(e)[:120])
            return None

    def _embed_texts_via_api(self, texts: list[str]) -> list[list[float]] | None:
        if not self._hf_token:
            return None
        try:
            client = InferenceClient(token=self._hf_token)
            embeddings: list[list[float]] = []
            for text in texts:
                result = client.feature_extraction(text, model=self._model_name)
                emb = result[0] if isinstance(result[0], list) else result
                embeddings.append(self._normalize(emb))
            return embeddings
        except Exception as e:
            log.warning("API batch embedding failed, trying local", error=str(e)[:120])
            return None

    def _load_local_model(self):
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                log.info("loading local embedding model", model=self._model_name)
                self._local_model = SentenceTransformer(self._model_name, device="cpu")
            except ImportError:
                log.warning("sentence-transformers not installed")
                raise
        return self._local_model

    def _embed_local(self, text: str) -> list[float]:
        model = self._load_local_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def _embed_texts_local(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        model = self._load_local_model()
        embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return embeddings.tolist()

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]
