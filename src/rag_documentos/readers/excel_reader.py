from __future__ import annotations

from typing import BinaryIO

import pandas as pd


def read_excel(file: BinaryIO) -> str:
    """Extrae las hojas de un libro Excel como texto tabular."""
    workbook = pd.ExcelFile(file)
    sheets = []

    for sheet_name in workbook.sheet_names:
        dataframe = pd.read_excel(workbook, sheet_name=sheet_name).fillna("")
        if dataframe.empty and len(dataframe.columns) == 0:
            continue
        content = dataframe.to_string(index=False)
        sheets.append(f"[Hoja: {sheet_name}]\n{content}")

    return "\n\n".join(sheets)
