from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_documentos.rag_service import get_collection  # noqa: E402
from rag_documentos.searching.global_search import search_documents  # noqa: E402


st.title("Búsqueda global")
st.write(
    "Describe la información que buscas. Se revisarán los documentos indexados "
    "y se mostrarán solo los que contengan información relacionada."
)

try:
    indexed_count = get_collection().count()
except Exception as error:
    st.error(f"No se pudo abrir el índice local: {error}")
    indexed_count = 0

with st.form("global-search-form"):
    search_query = st.text_input(
        "¿Qué información quieres localizar?",
        placeholder="Ej.: ¿En qué documento hay información sobre programación?",
    )
    submitted = st.form_submit_button("Buscar en todos los documentos")

if submitted:
    try:
        results = search_documents(search_query)
        if results:
            st.caption(f"Documentos encontrados: {len(results)}")
            for result in results:
                with st.expander(result.source):
                    st.write(result.excerpt)
        elif indexed_count:
            st.info("No se encontraron documentos con información relacionada.")
        else:
            st.info("No hay documentos indexados para buscar.")
    except Exception as error:
        st.error(f"No se pudo completar la búsqueda: {error}")
