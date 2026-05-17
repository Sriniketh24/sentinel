import gc

import structlog
from huggingface_hub import InferenceClient

from src.agents.state import PipelineState
from src.config import get_settings

log = structlog.get_logger()

DOC_TYPE_LABELS = [
    "annual financial report",
    "quarterly financial report",
    "material event disclosure",
    "earnings call transcript",
    "analyst presentation",
    "news article",
]

SENTIMENT_LABELS = ["bullish", "bearish", "neutral"]

CLASSIFIER_MODEL = "typeform/distilbert-base-uncased-mnli"


class ClassifierAgent:
    def run(self, state: PipelineState) -> PipelineState:
        text = state.content[:1500]

        result = self._classify_via_api(text)
        if result is None:
            result = self._classify_local(text)

        if result:
            state.doc_type = result["doc_type"]
            state.sentiment = result["sentiment"]
            state.sentiment_confidence = result["sentiment_confidence"]

        log.info(
            "classified",
            doc_type=state.doc_type,
            sentiment=state.sentiment,
            confidence=round(state.sentiment_confidence, 3),
        )
        return state

    @staticmethod
    def _classify_via_api(text: str) -> dict | None:
        settings = get_settings()
        if not settings.hf_api_token:
            return None
        try:
            client = InferenceClient(token=settings.hf_api_token)

            doc_result = client.zero_shot_classification(
                text, DOC_TYPE_LABELS, model=CLASSIFIER_MODEL
            )
            sent_result = client.zero_shot_classification(
                text, SENTIMENT_LABELS, model=CLASSIFIER_MODEL
            )

            return {
                "doc_type": doc_result[0]["label"],
                "sentiment": sent_result[0]["label"],
                "sentiment_confidence": sent_result[0]["score"],
            }
        except Exception as e:
            log.warning("API classification failed, trying local", error=str(e)[:120])
            return None

    @staticmethod
    def _classify_local(text: str) -> dict | None:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            log.warning("transformers not installed, skipping local classification")
            return {
                "doc_type": "unknown",
                "sentiment": "neutral",
                "sentiment_confidence": 0.5,
            }

        classifier = hf_pipeline(
            "zero-shot-classification",
            model=CLASSIFIER_MODEL,
            device=-1,
        )

        doc_result = classifier(text, DOC_TYPE_LABELS)
        sent_result = classifier(text, SENTIMENT_LABELS)

        result = {
            "doc_type": doc_result["labels"][0],
            "sentiment": sent_result["labels"][0],
            "sentiment_confidence": sent_result["scores"][0],
        }

        del classifier
        gc.collect()
        return result
