"""Tests for document loaders (PDF, DOCX, Web)."""

import pytest
from unittest.mock import patch, MagicMock
from app.core.exceptions import (
    ProcessingError,
    DocumentTooLargeError,
    ScrapingError,
)


class TestWebLoaderSecurity:
    """Tests for SSRF protection in the web loader."""

    def test_blocks_private_ip(self):
        from app.rag.loaders.web_loader import resolve_and_check

        # Patch socket.getaddrinfo to return a private IP
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("192.168.1.1", 443))
            ]
            with pytest.raises(ScrapingError, match="restricted IP"):
                resolve_and_check("https://evil.com")

    def test_blocks_loopback(self):
        from app.rag.loaders.web_loader import resolve_and_check

        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("127.0.0.1", 443))
            ]
            with pytest.raises(ScrapingError, match="restricted IP"):
                resolve_and_check("https://localhost")

    def test_blocks_metadata_ip(self):
        from app.rag.loaders.web_loader import resolve_and_check

        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("169.254.169.254", 80))
            ]
            with pytest.raises(ScrapingError, match="restricted IP"):
                resolve_and_check("http://metadata.internal")

    def test_blocks_invalid_scheme(self):
        from app.rag.loaders.web_loader import resolve_and_check

        with pytest.raises(ScrapingError, match="scheme"):
            resolve_and_check("ftp://example.com/file")

    def test_blocks_no_hostname(self):
        from app.rag.loaders.web_loader import resolve_and_check

        with pytest.raises(ScrapingError, match="No hostname"):
            resolve_and_check("https://")

    def test_allows_public_ip(self):
        from app.rag.loaders.web_loader import resolve_and_check

        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("93.184.216.34", 443))
            ]
            ip, port, hostname = resolve_and_check("https://example.com")
            assert ip == "93.184.216.34"
            assert port == 443
            assert hostname == "example.com"

    def test_dns_failure_raises(self):
        from app.rag.loaders.web_loader import resolve_and_check

        with patch("socket.getaddrinfo", side_effect=Exception("DNS failed")):
            with pytest.raises(ScrapingError, match="DNS resolution"):
                resolve_and_check("https://nonexistent.invalid")


class TestPDFLoader:
    @pytest.mark.asyncio
    async def test_rejects_encrypted_pdf(self, tmp_path):
        """Encrypted PDFs should raise ProcessingError."""
        # Create a mock encrypted PDF
        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.is_encrypted = True
            mock_reader_cls.return_value = mock_reader

            from app.rag.loaders.pdf_loader import load_pdf

            pdf_path = str(tmp_path / "encrypted.pdf")
            with open(pdf_path, "wb") as f:
                f.write(b"%PDF-1.4 fake content")

            with pytest.raises(ProcessingError, match="encrypted"):
                await load_pdf(pdf_path)


class TestDocxLoader:
    @pytest.mark.asyncio
    async def test_rejects_oversized_docx(self, tmp_path):
        """DOCX with > 10000 paragraphs should raise DocumentTooLargeError."""
        with patch("docx.Document") as mock_doc_cls:
            # Simulate 10001 paragraphs
            mock_para = MagicMock()
            mock_para.text = "Some text"
            mock_para.style.name = "Normal"

            mock_doc = MagicMock()
            mock_doc.paragraphs = [mock_para] * 10001
            mock_doc_cls.return_value = mock_doc

            from app.rag.loaders.docx_loader import load_docx

            docx_path = str(tmp_path / "large.docx")
            with open(docx_path, "wb") as f:
                f.write(b"fake")

            with pytest.raises(DocumentTooLargeError):
                await load_docx(docx_path)
