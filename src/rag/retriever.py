import structlog

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
        self._reranker = None
        self._reranker_available: bool | None = None

    def _load_reranker(self):
        if self._reranker_available is False:
            return None
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                self._reranker_available = True
            except ImportError:
                log.info("CrossEncoder unavailable, using embedding similarity only")
                self._reranker_available = False
                return None
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
        if reranker is None:
            # Fallback: use vector similarity score as rerank_score
            for c in candidates:
                c["rerank_score"] = c.get("score", 0.0)
            return candidates[: self._rerank_top_k]

        pairs = [(query, c["content"]) for c in candidates]
        scores = reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[: self._rerank_top_k]
