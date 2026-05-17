"""Tests for src.parsers.pdf module."""

import pytest

from src.parsers.pdf import extract_pdf_text


def _make_minimal_pdf(text: str = "Hello World") -> bytes:
    """Create a minimal valid PDF with the given text using pdfplumber-compatible format."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from io import BytesIO

        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        c.drawString(100, 700, text)
        c.showPage()
        c.save()
        return buf.getvalue()
    except ImportError:
        pytest.skip("reportlab not installed — skipping PDF generation test")


class TestExtractPdfText:
    def test_valid_pdf_returns_text(self):
        pdf_bytes = _make_minimal_pdf("Revenue was $10 billion")
        result = extract_pdf_text(pdf_bytes, filename="test.pdf")

        assert result["filename"] == "test.pdf"
        assert result["page_count"] == 1
        assert result["has_text"] is True
        assert result["character_count"] > 0
        assert "Revenue" in result["content"]
        assert "--- Page 1 ---" in result["content"]

    def test_filename_default(self):
        pdf_bytes = _make_minimal_pdf("Test content here")
        result = extract_pdf_text(pdf_bytes)

        assert result["filename"] == "upload.pdf"

    def test_invalid_bytes_returns_error(self):
        result = extract_pdf_text(b"not a pdf", filename="bad.pdf")

        assert result["has_text"] is False
        assert "error" in result
        assert result["page_count"] == 0

    def test_empty_pdf_returns_scanned_message(self):
        """An empty/image-only PDF should return the scanned fallback."""
        # Minimal PDF with no text content — just a blank page
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from io import BytesIO

            buf = BytesIO()
            c = canvas.Canvas(buf, pagesize=letter)
            c.showPage()  # blank page, no text
            c.save()
            pdf_bytes = buf.getvalue()
        except ImportError:
            pytest.skip("reportlab not installed")

        result = extract_pdf_text(pdf_bytes, filename="scanned.pdf")

        assert result["has_text"] is False
        assert "OCR" in result.get("error", "")

    def test_not_pdf_bytes(self):
        """Random bytes should fail gracefully."""
        result = extract_pdf_text(b"\x00\x01\x02\x03", filename="corrupt.pdf")

        assert result["has_text"] is False
        assert "error" in result
