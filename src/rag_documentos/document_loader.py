from __future__ import annotations

from typing import BinaryIO, Callable

from rag_documentos.readers.docx_reader import read_docx
from rag_documentos.readers.excel_reader import read_excel
from rag_documentos.readers.pdf_reader import read_pdf

Reader = Callable[[BinaryIO], str]

READERS: dict[str, Reader] = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".xls": read_excel,
    ".xlsx": read_excel,
}


def load_document(filename: str, file: BinaryIO) -> str:
    """Selecciona el lector según la extensión del archivo."""
    extension = filename.rpartition(".")[2].lower()
    if not extension:
        raise ValueError(f"El archivo '{filename}' no tiene extensión.")

    reader = READERS.get(f".{extension}")
    if reader is None:
        raise ValueError(f"Formato no compatible: .{extension}")

    file.seek(0)
    return reader(file)
