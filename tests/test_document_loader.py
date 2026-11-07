from io import BytesIO

import pandas as pd
import pytest
from docx import Document

from rag_documentos.document_loader import load_document


def test_load_docx_extracts_paragraphs_and_tables() -> None:
    document = Document()
    document.add_paragraph("Texto de prueba")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Nombre"
    table.cell(0, 1).text = "Valor"
    buffer = BytesIO()
    document.save(buffer)
    buffer.seek(0)

    text = load_document("ejemplo.DOCX", buffer)

    assert "Texto de prueba" in text
    assert "Nombre | Valor" in text


def test_load_document_rejects_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="Formato no compatible"):
        load_document("ejemplo.txt", BytesIO())


def test_load_xlsx_includes_sheet_name_and_rows() -> None:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"Producto": ["Café"], "Cantidad": [3]}).to_excel(
            writer, sheet_name="Ventas", index=False
        )
    buffer.seek(0)

    text = load_document("ejemplo.xlsx", buffer)

    assert "[Hoja: Ventas]" in text
    assert "Café" in text
    assert "3" in text
