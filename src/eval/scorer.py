import math
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


@dataclass
class GoldenExample:
    query: str
    expected_answer: str
    expected_entities: list[str]
    relevant_doc_ids: list[str]


class RetrievalScorer:
    @staticmethod
    def mrr(results: list[list[str]], ground_truth: list[list[str]]) -> float:
        if not results:
            return 0.0
        reciprocal_ranks = []
        for retrieved, relevant in zip(results, ground_truth):
            rr = 0.0
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    @staticmethod
    def ndcg_at_k(results: list[list[str]], ground_truth: list[list[str]], k: int = 10) -> float:
        if not results:
            return 0.0

        scores = []
        for retrieved, relevant in zip(results, ground_truth):
            dcg = 0.0
            for i, doc_id in enumerate(retrieved[:k]):
                if doc_id in relevant:
                    dcg += 1.0 / math.log2(i + 2)

            n_relevant = min(len(relevant), k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant)) if n_relevant > 0 else 0.0
            ndcg = min(dcg / idcg, 1.0) if idcg > 0 else 0.0
            scores.append(ndcg)

        return sum(scores) / len(scores)


class ExtractionScorer:
    @staticmethod
    def entity_f1(
        predicted: list[str],
        expected: list[str],
    ) -> tuple[float, float, float]:
        if not predicted and not expected:
            return 1.0, 1.0, 1.0
        if not predicted or not expected:
            return 0.0, 0.0, 0.0

        pred_set = {p.lower().strip() for p in predicted}
        exp_set = {e.lower().strip() for e in expected}

        tp = len(pred_set & exp_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(exp_set) if exp_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    @staticmethod
    def metric_accuracy(
        predicted: dict[str, str],
        expected: dict[str, str],
    ) -> float:
        if not expected:
            return 1.0 if not predicted else 0.0

        correct = 0
        for key, exp_val in expected.items():
            pred_val = predicted.get(key, "")
            if pred_val and _normalize_number(pred_val) == _normalize_number(exp_val):
                correct += 1

        return correct / len(expected)


def _normalize_number(s: str) -> str:
    import re

    nums = re.findall(r"[\d,.]+", s)
    if nums:
        return nums[0].replace(",", "")
    return s.strip().lower()
