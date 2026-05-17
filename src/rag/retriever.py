import structlog
from sentence_transformers import CrossEncoder

from src.config import get_settings
from src.rag.embedder import Embedder
from src.rag.store import VectorStore

log = structlog.get_logger()


class HybridRetriever:
    def __init__(self) -> None:
        settings = get_settings()
        self._embedder = Embedder()
        self._store = VectorStore()
        self._top_k = settings.top_k
        self._rerank_top_k = settings.rerank_top_k
        self._reranker: CrossEncoder | None = None

    def _load_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker

    def retrieve(self, query: str, modality_filter: str | None = None) -> list[dict]:
        query_embedding = self._embedder.embed_text(query)

        candidates = self._store.search(
            query_vector=query_embedding,
            top_k=self._top_k,
            modality_filter=modality_filter,
        )

        if not candidates:
            return []

        reranked = self._rerank(query, candidates)

        log.info(
            "retrieved",
            query_len=len(query),
            candidates=len(candidates),
            reranked=len(reranked),
        )
        return reranked

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        reranker = self._load_reranker()
        pairs = [(query, c["content"]) for c in candidates]
        scores = reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[: self._rerank_top_k]
