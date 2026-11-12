"""Unit tests for empty documents handling (R-20) and DOCX loader (R-21)."""

import pytest
import docx
from unittest.mock import patch, MagicMock, AsyncMock

from app.core.exceptions import ProcessingError
from app.rag.loaders.pdf_loader import load_pdf
from app.rag.loaders.docx_loader import load_docx
from app.rag.ingestion import ingestion_pipeline


class TestEmptyDocumentsR20:
    @pytest.mark.asyncio
    async def test_pdf_with_no_text_raises_processing_error(self, tmp_path):
        """R-20: PDF with no extractable text (e.g. image-only or empty) raises ProcessingError."""
        pdf_path = str(tmp_path / "empty.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy")

        with patch("pypdf.PdfReader") as mock_reader_cls, \
             patch("pdfplumber.open") as mock_plumber_open:

            mock_reader = MagicMock()
            mock_reader.is_encrypted = False
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "   \n\t  "  # Only whitespace
            mock_reader.pages = [mock_page]
            mock_reader_cls.return_value = mock_reader

            mock_plumber_pdf = MagicMock()
            mock_plumber_page = MagicMock()
            mock_plumber_page.extract_tables.return_value = []
            mock_plumber_pdf.pages = [mock_plumber_page]
            mock_plumber_open.return_value.__enter__.return_value = mock_plumber_pdf

            with pytest.raises(ProcessingError, match="No extractable text found"):
                await load_pdf(pdf_path)

    @pytest.mark.asyncio
    async def test_empty_docx_raises_processing_error(self, tmp_path):
        """R-20: Empty DOCX raises ProcessingError."""
        docx_path = str(tmp_path / "empty.docx")
        doc = docx.Document()
        doc.save(docx_path)

        with pytest.raises(ProcessingError, match="No extractable text found"):
            await load_docx(docx_path)

    @pytest.mark.asyncio
    async def test_ingestion_of_empty_document_ends_failed_without_storing(self, tmp_path):
        """R-20: When loader yields no text, job ends failed: No extractable text found and nothing is stored."""
        pdf_path = str(tmp_path / "blank.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 blank")

        with patch("app.rag.ingestion.load_pdf", side_effect=ProcessingError("No extractable text found")), \
             patch("app.rag.ingestion.set_cache", new_callable=AsyncMock) as mock_cache, \
             patch("app.rag.ingestion.add_documents", new_callable=AsyncMock) as mock_add_chroma:

            await ingestion_pipeline.process_file(
                pdf_path, "blank.pdf", "application/pdf", "job-blank-1", owner_id="1"
            )

            mock_cache.assert_any_call("job:job-blank-1", "failed: No extractable text found")
            mock_add_chroma.assert_not_called()


class TestDocxLoaderR21:
    @pytest.mark.asyncio
    async def test_docx_groups_paragraphs_under_headings_and_extracts_tables(self, tmp_path):
        """R-21: DOCX loader groups paragraphs under nearest heading and formats tables with | cells in order."""
        docx_path = str(tmp_path / "structured.docx")
        doc = docx.Document()

        # Section 1
        doc.add_heading("Overview", level=1)
        doc.add_paragraph("This is the first paragraph of overview.")
        doc.add_paragraph("This is the second paragraph of overview.")

        # Table inside Section 1
        table = doc.add_table(rows=2, cols=3)
        table.cell(0, 0).text = "Metric"
        table.cell(0, 1).text = "Value"
        table.cell(0, 2).text = "Unit"
        table.cell(1, 0).text = "Latency"
        table.cell(1, 1).text = "42"
        table.cell(1, 2).text = "ms"

        doc.add_paragraph("Concluding overview thoughts.")

        # Section 2
        doc.add_heading("Architecture Details", level=1)
        doc.add_paragraph("Details about the system design.")

        doc.save(docx_path)

        documents = await load_docx(docx_path)

        # Must produce 2 section-sized documents (not 1 per paragraph!)
        assert len(documents) == 2

        # Check section 1
        doc1 = documents[0]
        assert doc1.metadata["section"] == "Overview"
        assert "Overview" in doc1.page_content
        assert "This is the first paragraph of overview." in doc1.page_content
        assert "This is the second paragraph of overview." in doc1.page_content
        # Table rows joined by |
        assert "Metric | Value | Unit" in doc1.page_content
        assert "Latency | 42 | ms" in doc1.page_content
        assert "Concluding overview thoughts." in doc1.page_content

        # Verify document ordering in section 1:
        idx_p2 = doc1.page_content.index("second paragraph")
        idx_tbl = doc1.page_content.index("Latency | 42 | ms")
        idx_p3 = doc1.page_content.index("Concluding overview thoughts.")
        assert idx_p2 < idx_tbl < idx_p3

        # Check section 2
        doc2 = documents[1]
        assert doc2.metadata["section"] == "Architecture Details"
        assert "Details about the system design." in doc2.page_content

    @pytest.mark.asyncio
    async def test_docx_without_headings_defaults_to_general_section(self, tmp_path):
        """R-21: DOCX without headings is grouped into a 'General' section."""
        docx_path = str(tmp_path / "plain.docx")
        doc = docx.Document()
        doc.add_paragraph("First plain paragraph.")
        doc.add_paragraph("Second plain paragraph.")
        doc.save(docx_path)

        documents = await load_docx(docx_path)
        assert len(documents) == 1
        assert documents[0].metadata["section"] == "General"
        assert "First plain paragraph." in documents[0].page_content
        assert "Second plain paragraph." in documents[0].page_content
