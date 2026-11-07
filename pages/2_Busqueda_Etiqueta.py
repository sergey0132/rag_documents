from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_documentos.tagging.document_tags import (  # noqa: E402
    list_existing_tags,
    list_untagged_documents,
    search_documents_by_tag,
)


st.title("Búsqueda por etiqueta")
st.write(
    "Describe qué tipo de documento buscas. La IA relacionará tu petición "
    "con una etiqueta existente y mostrará todos los documentos coincidentes."
)

try:
    existing_tags = list_existing_tags()
    if existing_tags:
        st.caption("Etiquetas disponibles: " + " · ".join(existing_tags))
    else:
        st.info("Todavía no hay etiquetas. Procesa o etiqueta documentos primero.")
except Exception as error:
    st.error(f"No se pudieron cargar las etiquetas disponibles: {error}")
    existing_tags = []

with st.form("tag-search-form"):
    tag_query = st.text_input(
        "¿Qué tipo de documento quieres encontrar?",
        placeholder="Ej.: Necesito los currículums de las personas candidatas",
    )
    submitted = st.form_submit_button("Buscar por etiqueta")

if submitted:
    try:
        results = search_documents_by_tag(tag_query)
        if results:
            st.caption(
                f"Documentos encontrados con la etiqueta "
                f"«{results[0].tag}»: {len(results)}"
            )
            for document in results:
                with st.expander(document.name):
                    st.caption(f"Etiqueta: {document.tag}")
                    st.write(document.excerpt or "No hay extracto disponible.")
        elif list_untagged_documents():
            st.info(
                "No se encontraron coincidencias. Puede que haya documentos "
                "pendientes de etiquetar."
            )
        else:
            st.info("No hay documentos con una etiqueta que coincida.")
    except Exception as error:
        st.error(f"No se pudo buscar por etiqueta: {error}")
