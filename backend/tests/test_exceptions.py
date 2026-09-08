"""Tests for custom exception hierarchy."""

from app.core.exceptions import (
    ProcessingError,
    DocumentTooLargeError,
    UnsupportedFormatError,
    ScrapingError,
)


class TestExceptionHierarchy:
    def test_document_too_large_is_processing_error(self):
        assert issubclass(DocumentTooLargeError, ProcessingError)

    def test_unsupported_format_is_processing_error(self):
        assert issubclass(UnsupportedFormatError, ProcessingError)

    def test_scraping_error_is_processing_error(self):
        assert issubclass(ScrapingError, ProcessingError)

    def test_processing_error_is_exception(self):
        assert issubclass(ProcessingError, Exception)

    def test_exception_messages(self):
        exc = DocumentTooLargeError("File too big: 50MB")
        assert "50MB" in str(exc)

        exc = UnsupportedFormatError("Cannot handle .xyz")
        assert ".xyz" in str(exc)
