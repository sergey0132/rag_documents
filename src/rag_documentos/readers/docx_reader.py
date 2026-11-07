from __future__ import annotations

from typing import BinaryIO

from docx import Document


def read_docx(file: BinaryIO) -> str:
    """Extrae párrafos y contenido de tablas de un documento DOCX."""
    document = Document(file)
    blocks = [paragraph.text.strip() for paragraph in document.paragraphs]

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            blocks.append(" | ".join(cells))

    return "\n".join(block for block in blocks if block)
