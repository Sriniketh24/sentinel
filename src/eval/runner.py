import json
import time
from pathlib import Path

import numpy as np
import structlog

from src.agents.graph import run_pipeline
from src.eval.metrics import EvalMetrics
from src.eval.scorer import ExtractionScorer, GoldenExample, RetrievalScorer
from src.rag.retriever import HybridRetriever

log = structlog.get_logger()


class EvalRunner:
    def __init__(self, golden_path: str = "data/eval/golden.json") -> None:
        self._golden_path = Path(golden_path)
        self._retrieval_scorer = RetrievalScorer()
        self._extraction_scorer = ExtractionScorer()
        self._retriever = HybridRetriever()

    def load_golden_set(self) -> list[GoldenExample]:
        if not self._golden_path.exists():
            log.warning("golden set not found", path=str(self._golden_path))
            return []

        with self._golden_path.open() as f:
            data = json.load(f)

        return [
            GoldenExample(
                query=item["query"],
                expected_answer=item.get("expected_answer", ""),
                expected_entities=item.get("expected_entities", []),
                relevant_doc_ids=item.get("relevant_doc_ids", []),
            )
            for item in data
        ]

    def run(self) -> EvalMetrics:
        examples = self.load_golden_set()
        if not examples:
            log.warning("no golden examples, returning empty metrics")
            return EvalMetrics()

        latencies: list[float] = []
        all_retrieved: list[list[str]] = []
        all_relevant: list[list[str]] = []
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []

        for example in examples:
            start = time.time()

            # Run retrieval separately to score it
            retrieved_docs = self._retriever.retrieve(example.query)
            retrieved_ids = [doc.get("document_id", "") for doc in retrieved_docs]
            all_retrieved.append(retrieved_ids)
            all_relevant.append(example.relevant_doc_ids)

            # Run full pipeline
            result = run_pipeline(
                document_id="eval",
                content=example.query,
                company_ticker="EVAL",
            )
            elapsed = time.time() - start
            latencies.append(elapsed)

            if result.extraction:
                pred_entities = [e.text for e in result.extraction.entities]
                p, r, f1 = self._extraction_scorer.entity_f1(
                    pred_entities, example.expected_entities
                )
                precisions.append(p)
                recalls.append(r)
                f1s.append(f1)

        retrieval_mrr = self._retrieval_scorer.mrr(all_retrieved, all_relevant)
        retrieval_ndcg = self._retrieval_scorer.ndcg_at_k(all_retrieved, all_relevant)

        metrics = EvalMetrics(
            retrieval_mrr=retrieval_mrr,
            retrieval_ndcg=retrieval_ndcg,
            extraction_precision=float(np.mean(precisions)) if precisions else 0.0,
            extraction_recall=float(np.mean(recalls)) if recalls else 0.0,
            extraction_f1=float(np.mean(f1s)) if f1s else 0.0,
            latency_p50=float(np.percentile(latencies, 50)) if latencies else 0.0,
            latency_p95=float(np.percentile(latencies, 95)) if latencies else 0.0,
        )

        log.info("eval complete", **metrics.summary())
        return metrics
