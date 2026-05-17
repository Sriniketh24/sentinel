from src.models.document import Document, DocumentChunk, DocumentType, Modality
from src.models.pipeline import ExtractionResult, PipelineResult, ResearchBrief


def test_document_defaults():
    doc = Document()
    assert doc.doc_type == DocumentType.UNKNOWN
    assert doc.company_ticker == ""


def test_document_chunk_modality():
    chunk = DocumentChunk(modality=Modality.TABLE)
    assert chunk.modality == Modality.TABLE
    assert chunk.embedding is None


def test_pipeline_result():
    result = PipelineResult(document_id="test_doc")
    assert result.total_tokens_used == 0
    assert result.extraction is None


def test_extraction_result():
    ext = ExtractionResult(
        document_id="d1",
        key_metrics={"revenue": "$100M"},
        claims=["We expect growth"],
    )
    assert len(ext.claims) == 1
    assert "revenue" in ext.key_metrics


def test_research_brief():
    brief = ResearchBrief(
        company_ticker="AAPL",
        title="Apple Q4 Analysis",
        key_findings=["Revenue grew 10%"],
    )
    assert brief.company_ticker == "AAPL"
    assert len(brief.key_findings) == 1
