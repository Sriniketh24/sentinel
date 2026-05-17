from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    text: str
    label: str
    confidence: float = 0.0


class ExtractionResult(BaseModel):
    document_id: str
    doc_type: str = ""
    sentiment: str = ""
    sentiment_confidence: float = 0.0
    entities: list[ExtractedEntity] = Field(default_factory=list)
    key_metrics: dict[str, str] = Field(default_factory=dict)
    claims: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    claim: str
    verified: bool = False
    confidence: float = 0.0
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)


class ResearchBrief(BaseModel):
    company_ticker: str
    title: str = ""
    summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    financial_highlights: dict[str, str] = Field(default_factory=dict)
    citations: list[dict[str, str]] = Field(default_factory=list)
    confidence_score: float = 0.0


class PipelineResult(BaseModel):
    document_id: str
    extraction: ExtractionResult | None = None
    verifications: list[VerificationResult] = Field(default_factory=list)
    brief: ResearchBrief | None = None
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    latency_seconds: float = 0.0
