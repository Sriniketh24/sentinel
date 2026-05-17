from typing import Any

from pydantic import BaseModel, Field

from src.models.pipeline import ExtractionResult, ResearchBrief, VerificationResult


class PipelineState(BaseModel):
    document_id: str = ""
    content: str = ""
    modality: str = "text"
    company_ticker: str = ""

    # Classifier output
    doc_type: str = ""
    sentiment: str = ""
    sentiment_confidence: float = 0.0

    # Extractor output
    extraction: ExtractionResult | None = None

    # Verifier output
    verifications: list[VerificationResult] = Field(default_factory=list)

    # Synthesizer output
    brief: ResearchBrief | None = None

    # Observability
    tokens_used: int = 0
    cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
