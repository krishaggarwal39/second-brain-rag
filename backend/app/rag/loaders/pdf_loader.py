import asyncio
import hashlib
from typing import List
import pypdf
import pdfplumber
from langchain_core.documents import Document
from app.core.exceptions import ProcessingError, DocumentTooLargeError

def _table_to_text(table) -> str:
    """Render a pdfplumber table as readable rows ("a | b | c"), not a Python repr (R-48)."""
    rows = []
    for row in table or []:
        cells = [" ".join(str(c).split()) if c is not None else "" for c in row or []]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _process_pdf_sync(file_path: str) -> List[Document]:
    documents = []
    
    with open(file_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        
        if reader.is_encrypted:
            raise ProcessingError("Cannot process encrypted PDF.")
            
        total_pages = len(reader.pages)
        if total_pages > 500:
            raise DocumentTooLargeError(f"PDF exceeds the maximum allowed page limit of 500 pages (found {total_pages} pages).")
            
        # PERFORMANCE FIX: Open pdfplumber ONCE outside the loop
        with pdfplumber.open(file_path) as pdf:
            for i in range(total_pages):
                page = reader.pages[i]
                text = page.extract_text() or ""
                
                # Extract tables using the already open pdfplumber instance
                plumber_page = pdf.pages[i]
                tables = plumber_page.extract_tables()
                if tables:
                    text += "\n" + "\n\n".join(_table_to_text(t) for t in tables if t)
            
                # R-20: Skip pages with no extractable text (e.g. blank or image-only pages)
                if not text.strip():
                    continue

                # Generate hash
                page_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                
                metadata = {
                    "filename": file_path.split("/")[-1],
                    "page_number": i + 1,
                    "total_pages": total_pages,
                    "source_type": "pdf",
                    "hash": page_hash
                }
                documents.append(Document(page_content=text, metadata=metadata))
            
    # R-20: If extraction yields no text (empty or image-only PDF), raise permanent ProcessingError
    if not documents:
        raise ProcessingError("No extractable text found")

    return documents

async def load_pdf(file_path: str) -> List[Document]:
    """Asynchronously load a PDF and extract its text and tables."""
    return await asyncio.to_thread(_process_pdf_sync, file_path)
