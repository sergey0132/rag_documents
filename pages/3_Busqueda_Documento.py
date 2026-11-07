from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_documentos.rag_service import (  # noqa: E402
    ask_document_question,
    list_indexed_documents,
)


st.title("Búsqueda por documento")
st.write(
    "Elige un archivo indexado y pregunta sobre su contenido. "
    "La recuperación se limitará al documento seleccionado."
)

try:
    indexed_documents = list_indexed_documents()
except Exception as error:
    st.error(f"No se pudo cargar la lista de documentos: {error}")
    indexed_documents = []

if indexed_documents:
    document_name_filter = st.text_input(
        "Buscar documento por nombre",
        placeholder="Escribe parte del nombre, por ejemplo: contrato",
    )
    matching_documents = [
        name
        for name in indexed_documents
        if document_name_filter.casefold() in name.casefold()
    ]

    if matching_documents:
        with st.form("document-question-form"):
            selected_document = st.selectbox(
                "Selecciona un documento",
                options=matching_documents,
            )
            question = st.text_input("¿Qué quieres saber de este documento?")
            submitted = st.form_submit_button("Consultar documento")

        if submitted:
            try:
                result = ask_document_question(selected_document, question)
                st.markdown(result.answer)
                with st.expander("Fragmentos consultados"):
                    for source in result.sources:
                        st.markdown(f"**{source.source}**")
                        st.write(source.text)
            except Exception as error:
                st.error(f"No se pudo consultar el documento: {error}")
    else:
        st.info("No hay documentos cuyo nombre coincida con esa búsqueda.")
else:
    st.info("Todavía no hay documentos indexados.")
