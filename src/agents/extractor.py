import gc
import re

import structlog
from huggingface_hub import InferenceClient

from src.agents.state import PipelineState
from src.config import get_settings
from src.models.pipeline import ExtractedEntity, ExtractionResult

log = structlog.get_logger()

NER_MODEL = "dslim/bert-base-NER"

FINANCIAL_PATTERNS = {
    "revenue": (
        r"(?:revenue|net\s+sales|total\s+revenue)"
        r"\s*(?:was|of|:)?\s*\$?([\d,.]+)\s*(million|billion|M|B)?"
    ),
    "net_income": (
        r"(?:net\s+income|net\s+earnings)"
        r"\s*(?:was|of|:)?\s*\$?([\d,.]+)\s*(million|billion|M|B)?"
    ),
    "eps": (
        r"(?:earnings\s+per\s+share|EPS)"
        r"\s*(?:was|were|of|:)?\s*\$?([\d.]+)"
    ),
    "gross_margin": (
        r"(?:gross\s+margin|gross\s+profit\s+margin)"
        r"\s*(?:was|of|:)?\s*([\d.]+)\s*%?(?:\s*percent)?"
    ),
}


class ExtractorAgent:
    def run(self, state: PipelineState) -> PipelineState:
        text = state.content

        entities = self._extract_entities(text)
        metrics = self._extract_financial_metrics(text)
        claims = self._extract_claims(text)

        state.extraction = ExtractionResult(
            document_id=state.document_id,
            doc_type=state.doc_type,
            sentiment=state.sentiment,
            sentiment_confidence=state.sentiment_confidence,
            entities=entities,
            key_metrics=metrics,
            claims=claims,
        )

        log.info(
            "extracted",
            entities=len(entities),
            metrics=len(metrics),
            claims=len(claims),
        )
        return state

    def _extract_entities(self, text: str) -> list[ExtractedEntity]:
        entities = self._ner_via_api(text)
        if entities is None:
            entities = self._ner_local(text)
        return entities

    @staticmethod
    def _ner_via_api(text: str) -> list[ExtractedEntity] | None:
        settings = get_settings()
        if not settings.hf_api_token:
            return None
        try:
            client = InferenceClient(token=settings.hf_api_token)
            raw = client.token_classification(text[:3000], model=NER_MODEL)

            seen: set[tuple[str, str]] = set()
            entities: list[ExtractedEntity] = []
            for ent in raw:
                word = ent.get("word", ent.get("entity", ""))
                label = ent.get("entity_group", ent.get("entity", ""))
                # Strip B-/I- prefixes from BIO tags
                if label.startswith(("B-", "I-")):
                    label = label[2:]
                key = (word, label)
                if key not in seen:
                    seen.add(key)
                    entities.append(
                        ExtractedEntity(
                            text=word,
                            label=label,
                            confidence=ent.get("score", 0.0),
                        )
                    )
            return entities
        except Exception as e:
            log.warning("API NER failed, trying local", error=str(e)[:120])
            return None

    @staticmethod
    def _ner_local(text: str) -> list[ExtractedEntity]:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            log.warning("transformers not installed, skipping NER")
            return []

        ner = hf_pipeline("ner", model=NER_MODEL, aggregation_strategy="simple", device=-1)
        raw = ner(text[:3000])

        seen: set[tuple[str, str]] = set()
        entities: list[ExtractedEntity] = []
        for ent in raw:
            key = (ent["word"], ent["entity_group"])
            if key not in seen:
                seen.add(key)
                entities.append(
                    ExtractedEntity(
                        text=ent["word"],
                        label=ent["entity_group"],
                        confidence=ent["score"],
                    )
                )

        del ner
        gc.collect()
        return entities

    @staticmethod
    def _extract_financial_metrics(text: str) -> dict[str, str]:
        metrics: dict[str, str] = {}
        for name, pattern in FINANCIAL_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                unit = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
                metrics[name] = f"${value} {unit}".strip()
        return metrics

    @staticmethod
    def _extract_claims(text: str) -> list[str]:
        claim_indicators = [
            r"(?:we\s+expect|management\s+expects|the\s+company\s+(?:expects|anticipates))",
            r"(?:guidance\s+(?:of|for|is))",
            r"(?:we\s+believe|we\s+project|management\s+believes|the\s+company\s+believes)",
            r"(?:year-over-year\s+(?:growth|decline|increase|decrease))",
            r"(?:expects?\s+continued|return\s+to\s+growth|outlook\s+(?:for|is|remains))",
            r"(?:we\s+(?:remain|are)\s+(?:confident|optimistic|cautious))",
        ]
        claims: list[str] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            for pattern in claim_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    cleaned = sentence.strip()
                    if 20 < len(cleaned) < 500:
                        claims.append(cleaned)
                    break
        return claims[:20]
