import gc

import structlog
from transformers import pipeline as hf_pipeline

from src.agents.state import PipelineState

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
        classifier = hf_pipeline(
            "zero-shot-classification",
            model=CLASSIFIER_MODEL,
            device=-1,
        )
        text = state.content[:1500]

        doc_result = classifier(text, DOC_TYPE_LABELS)
        state.doc_type = doc_result["labels"][0]

        sent_result = classifier(text, SENTIMENT_LABELS)
        state.sentiment = sent_result["labels"][0]
        state.sentiment_confidence = sent_result["scores"][0]

        log.info(
            "classified",
            doc_type=state.doc_type,
            sentiment=state.sentiment,
            confidence=round(state.sentiment_confidence, 3),
        )

        del classifier
        gc.collect()
        return state
