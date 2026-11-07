from __future__ import annotations

from typing import BinaryIO

from pypdf import PdfReader


def read_pdf(file: BinaryIO) -> str:
    """Extrae el texto de todas las páginas de un PDF."""
    reader = PdfReader(file)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Página {page_number}]\n{text.strip()}")
    return "\n\n".join(pages)
