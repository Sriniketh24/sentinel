from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    SEC_10K = "10-K"
    SEC_10Q = "10-Q"
    SEC_8K = "8-K"
    EARNINGS_CALL = "earnings_call"
    SLIDE_DECK = "slide_deck"
    NEWS_ARTICLE = "news_article"
    ANALYST_REPORT = "analyst_report"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    AUDIO = "audio"


class Document(BaseModel):
    id: str = Field(default_factory=lambda: "")
    source_url: str = ""
    file_path: str = ""
    doc_type: DocumentType = DocumentType.UNKNOWN
    company_ticker: str = ""
    company_name: str = ""
    filing_date: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DocumentChunk(BaseModel):
    id: str = ""
    document_id: str = ""
    content: str = ""
    modality: Modality = Modality.TEXT
    chunk_index: int = 0
    page_number: int | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
