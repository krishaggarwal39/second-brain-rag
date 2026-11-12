import asyncio
import hashlib
from typing import List
import docx
from langchain_core.documents import Document
from app.core.exceptions import ProcessingError, DocumentTooLargeError


def _process_docx_sync(file_path: str) -> List[Document]:
    doc = docx.Document(file_path)

    # Check limit against memory exhaustion (DoS)
    # Paragraph count + table row count
    paragraphs = getattr(doc, "paragraphs", [])
    tables = getattr(doc, "tables", [])
    total_elements = len(paragraphs)
    for table in tables:
        total_elements += len(getattr(table, "rows", []))
    if total_elements > 10000:
        raise DocumentTooLargeError(
            "DOCX exceeds the maximum allowed limit of 10,000 elements. This is to prevent memory exhaustion (DoS)."
        )

    documents: List[Document] = []
    current_section = "General"
    current_block: List[str] = []
    filename = file_path.split("/")[-1]

    def _flush_section():
        nonlocal current_block, current_section
        content = "\n\n".join(t for t in current_block if t.strip()).strip()
        if content:
            doc_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            metadata = {
                "filename": filename,
                "section": current_section,
                "source_type": "docx",
                "hash": doc_hash,
            }
            documents.append(Document(page_content=content, metadata=metadata))
        current_block = []

    # R-21: Preserve document order by iterating through body elements
    body_elements = getattr(getattr(doc, "element", None), "body", None)
    if body_elements is not None:
        from docx.text.paragraph import Paragraph
        from docx.table import Table

        for child in body_elements:
            if child.tag.endswith("p"):
                para = Paragraph(child, doc)
                text = para.text.strip()
                if not text:
                    continue

                style_name = para.style.name if para.style and hasattr(para.style, "name") else ""
                if style_name.startswith("Heading"):
                    _flush_section()
                    current_section = text
                    current_block.append(text)
                else:
                    current_block.append(text)

            elif child.tag.endswith("tbl"):
                table = Table(child, doc)
                table_lines = []
                for row in table.rows:
                    row_text = " | ".join(c.text.strip() for c in row.cells)
                    if row_text.strip():
                        table_lines.append(row_text)
                if table_lines:
                    current_block.append("\n".join(table_lines))
    else:
        # Fallback if element.body is unavailable (e.g. minimal mock in tests)
        for para in paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style_name = para.style.name if para.style and hasattr(para.style, "name") else ""
            if style_name.startswith("Heading"):
                _flush_section()
                current_section = text
                current_block.append(text)
            else:
                current_block.append(text)

    # Flush any remaining content in the last section
    _flush_section()

    # R-20: If extraction yields no text, raise permanent ProcessingError
    if not documents:
        raise ProcessingError("No extractable text found")

    return documents


async def load_docx(file_path: str) -> List[Document]:
    """Asynchronously load a docx and extract section-sized documents with tables."""
    return await asyncio.to_thread(_process_docx_sync, file_path)
