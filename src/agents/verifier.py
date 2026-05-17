import structlog

from src.agents.state import PipelineState
from src.models.pipeline import VerificationResult
from src.rag.retriever import HybridRetriever

log = structlog.get_logger()

SUPPORT_THRESHOLD = 0.65
CONTRADICT_THRESHOLD = 0.3


class VerifierAgent:
    def __init__(self) -> None:
        self._retriever = HybridRetriever()

    def run(self, state: PipelineState) -> PipelineState:
        if not state.extraction or not state.extraction.claims:
            log.info("no claims to verify")
            return state

        verifications: list[VerificationResult] = []
        for claim in state.extraction.claims:
            result = self._verify_claim(claim)
            verifications.append(result)

        state.verifications = verifications
        verified_count = sum(1 for v in verifications if v.verified)
        log.info(
            "verified claims",
            total=len(verifications),
            verified=verified_count,
        )
        return state

    def _verify_claim(self, claim: str) -> VerificationResult:
        evidence = self._retriever.retrieve(claim)

        supporting: list[str] = []
        contradicting: list[str] = []

        for doc in evidence:
            score = doc.get("rerank_score", 0)
            content = doc.get("content", "")[:200]
            if score >= SUPPORT_THRESHOLD:
                supporting.append(content)
            elif score <= CONTRADICT_THRESHOLD:
                contradicting.append(content)

        verified = len(supporting) > 0 and len(supporting) > len(contradicting)
        confidence = max(doc.get("rerank_score", 0) for doc in evidence) if evidence else 0.0

        return VerificationResult(
            claim=claim,
            verified=verified,
            confidence=confidence,
            supporting_evidence=supporting[:3],
            contradicting_evidence=contradicting[:3],
        )
