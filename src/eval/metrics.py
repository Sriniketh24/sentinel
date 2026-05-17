import time
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


@dataclass
class EvalMetrics:
    retrieval_mrr: float = 0.0
    retrieval_ndcg: float = 0.0
    extraction_precision: float = 0.0
    extraction_recall: float = 0.0
    extraction_f1: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    total_cost_usd: float = 0.0
    eval_timestamp: float = field(default_factory=time.time)

    def summary(self) -> dict[str, float]:
        return {
            "retrieval_mrr": round(self.retrieval_mrr, 4),
            "retrieval_ndcg": round(self.retrieval_ndcg, 4),
            "extraction_f1": round(self.extraction_f1, 4),
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "latency_p50_s": round(self.latency_p50, 3),
            "latency_p95_s": round(self.latency_p95, 3),
            "total_cost_usd": round(self.total_cost_usd, 4),
        }
