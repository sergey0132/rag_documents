from __future__ import annotations

import hashlib
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_documentos.document_loader import load_document  # noqa: E402
from rag_documentos.rag_service import (  # noqa: E402
    index_document,
    list_indexed_documents,
)
from rag_documentos.tagging.document_tags import (  # noqa: E402
    CATEGORIES,
    DocumentClassification,
    classify_document,
    get_document_tags,
    list_untagged_documents,
    normalize_custom_tag,
    tag_unlabeled_documents,
    update_document_tag,
)

LOGO_PATH = PROJECT_ROOT / "assets" / "rag_logo_banner.png"

# Parámetros ajustables:
# - st.columns([1, 4, 1]): cambia los pesos para ensanchar o estrechar el área centrada.
# - width: usa "stretch" para ocuparla entera o un número de píxeles para fijarla.
# - caption: añade texto debajo del banner; None lo deja sin pie.
_, logo_column, _ = st.columns([1, 1, 1])
with logo_column:
    st.image(
        str(LOGO_PATH),
        width="stretch",
        caption=None,
        output_format="PNG",
        alt="Logo RAG de documentos",
    )

st.title("Gestión y subida de documentos")
st.write(
    "Sube documentos nuevos, revisa su clasificación o cambia las etiquetas "
    "de los archivos que ya están indexados."
)

tag_column, upload_column = st.columns([1, 2])

with tag_column:
    with st.container(border=True):
        st.subheader("Cambiar etiqueta")
        st.write("Actualiza el tag de un documento ya indexado.")
        try:
            documents_for_tag_edit = list_indexed_documents()
        except Exception as error:
            st.error(f"No se pudieron cargar los documentos indexados: {error}")
            documents_for_tag_edit = []

        if documents_for_tag_edit:
            selected_document = st.selectbox(
                "Documento",
                options=documents_for_tag_edit,
                key="tag-edit-document",
            )
            try:
                current_tags = get_document_tags(selected_document)
                st.caption("Etiqueta actual: " + (", ".join(current_tags) or "sin etiqueta"))
            except Exception as error:
                st.error(
                    f"No se pudo cargar la etiqueta de «{selected_document}»: {error}"
                )

            with st.form("edit-document-tag-form"):
                replacement_tag = st.text_input("Nueva etiqueta", max_chars=50)
                update_submitted = st.form_submit_button("Guardar etiqueta")

            if update_submitted:
                try:
                    updated_chunks = update_document_tag(
                        selected_document,
                        replacement_tag,
                    )
                    st.success(
                        f"Etiqueta actualizada en {updated_chunks} fragmentos "
                        f"de «{selected_document}»."
                    )
                except Exception as error:
                    st.error(f"No se pudo cambiar la etiqueta: {error}")
        else:
            st.info("Indexa un documento para poder cambiar su etiqueta.")

        try:
            untagged_documents = list_untagged_documents()
        except Exception as error:
            st.error(f"No se pudo comprobar el etiquetado de los documentos: {error}")
            untagged_documents = []

        if untagged_documents:
            with st.expander("Etiquetar documentos indexados anteriormente"):
                st.write(
                    "Clasifica archivos existentes que aún no tienen etiqueta, "
                    "sin volver a subirlos."
                )
                st.caption(f"Documentos pendientes: {len(untagged_documents)}")
                if st.button("Generar etiquetas faltantes"):
                    try:
                        updated_tags = tag_unlabeled_documents()
                        for name, tag in updated_tags.items():
                            st.success(f"{name}: etiqueta «{tag}».")
                        if not updated_tags:
                            st.info("No quedan documentos pendientes de etiquetar.")
                    except Exception as error:
                        st.error(f"No se pudieron etiquetar los documentos: {error}")

with upload_column:
    st.subheader("Subir documentos")
    upload_batch_id = st.session_state.setdefault("upload_batch_id", 0)
    files = st.file_uploader(
        "Selecciona uno o varios documentos",
        type=["pdf", "docx", "xlsx", "xls"],
        accept_multiple_files=True,
        # Cambiar la key tras indexar crea un uploader nuevo y limpia la selección.
        key=f"document-uploader-{upload_batch_id}",
    )

    if files:
        st.caption(
            "Clasifica los documentos con IA, revisa las propuestas y luego "
            "indexa los archivos seleccionados."
        )
        st.caption("Categorías disponibles: " + " · ".join(CATEGORIES))
        upload_classifications = st.session_state.setdefault(
            "upload_classifications", {}
        )
        file_records = []
        for uploaded_file in files:
            content = uploaded_file.getvalue()
            digest = hashlib.sha256(
                uploaded_file.name.encode("utf-8") + b"\0" + content
            ).hexdigest()
            file_records.append((uploaded_file, content, digest))

        if st.button("Clasificar documentos con IA", key="classify-uploads"):
            for uploaded_file, content, digest in file_records:
                try:
                    # Clasifica por el texto extraído, no por el nombre ni la
                    # extensión; así un PDF puede quedar como «Libro».
                    text = load_document(uploaded_file.name, BytesIO(content))
                    result = classify_document(text)
                    upload_classifications[digest] = {
                        "category": result.category,
                        "subcategory": result.subcategory,
                    }
                    st.success(f"{uploaded_file.name}: clasificación completada.")
                except Exception as error:
                    upload_classifications.pop(digest, None)
                    st.error(f"No se pudo clasificar {uploaded_file.name}: {error}")

        classifications_ready = all(
            digest in upload_classifications
            for _, _, digest in file_records
        )
        custom_tags = {}

        for uploaded_file, content, digest in file_records:
            classification_data = upload_classifications.get(digest)
            with st.expander(f"Documento: {uploaded_file.name}", expanded=True):
                if classification_data:
                    # La propuesta se puede corregir antes de guardarla en Chroma.
                    category = st.selectbox(
                        "Categoría (puedes corregir la propuesta de la IA)",
                        options=CATEGORIES,
                        index=CATEGORIES.index(classification_data["category"]),
                        key=f"upload-category-{upload_batch_id}-{digest}",
                    )
                    st.text_input(
                        "Subcategoría",
                        value=classification_data["subcategory"],
                        key=f"upload-subcategory-{upload_batch_id}-{digest}",
                    )
                    if category == "Otro":
                        # Si se deja vacío, la etiqueta almacenada será «Otro».
                        custom_tags[digest] = st.text_input(
                            "Etiqueta personalizada (opcional)",
                            placeholder="Ej.: Partitura, receta, documento histórico",
                            key=f"custom-tag-{upload_batch_id}-{digest}",
                        )

                if st.checkbox(
                    "Mostrar vista previa del texto",
                    key=f"show-preview-{upload_batch_id}-{digest}",
                ):
                    try:
                        text = load_document(uploaded_file.name, BytesIO(content))
                        st.text_area(
                            "Texto extraído",
                            value=text,
                            height=300,
                            key=f"extracted-{digest}",
                        )
                    except Exception as error:
                        st.error(f"No se pudo leer {uploaded_file.name}: {error}")

        if st.button(
            "Indexar documentos clasificados",
            type="primary",
            key="index-classified-uploads",
            disabled=not classifications_ready,
        ):
            all_documents_indexed = True
            for uploaded_file, content, digest in file_records:
                category = st.session_state[
                    f"upload-category-{upload_batch_id}-{digest}"
                ]
                subcategory = st.session_state[
                    f"upload-subcategory-{upload_batch_id}-{digest}"
                ]
                try:
                    classification = DocumentClassification(
                        category=category,
                        subcategory=subcategory,
                    )
                    custom_tag = custom_tags.get(digest, "").strip() or None
                    if custom_tag:
                        custom_tag = normalize_custom_tag(custom_tag)
                    chunks = index_document(
                        uploaded_file.name,
                        content,
                        classification=classification,
                        custom_tag=custom_tag,
                    )
                    displayed_tag = custom_tag or category
                    st.success(
                        f"{uploaded_file.name}: indexado como «{displayed_tag}» "
                        f"en {chunks} fragmentos."
                    )
                except Exception as error:
                    all_documents_indexed = False
                    st.error(f"No se pudo indexar {uploaded_file.name}: {error}")

            if all_documents_indexed:
                # Limpiar el estado de clasificación y cambiar la key del uploader
                # deja la página preparada para el siguiente lote de archivos.
                for _, _, digest in file_records:
                    upload_classifications.pop(digest, None)
                st.session_state["upload_batch_id"] = upload_batch_id + 1
                st.rerun()

st.divider()
st.subheader("¿Qué hacer después?")

try:
    existing_documents = list_indexed_documents()
except Exception as error:
    st.error(f"No se pudo consultar el índice de documentos: {error}")
    existing_documents = []

if existing_documents:
    st.success(
        f"Ya hay {len(existing_documents)} documento(s) indexado(s). "
        "No hace falta volver a subirlos: puedes ir directamente a una búsqueda."
    )
else:
    st.info(
        "Cuando termines de subir e indexar tus archivos, elige el método de "
        "búsqueda en la barra lateral."
    )

st.write(
    "Usa la barra lateral para abrir **Búsqueda global**, **Búsqueda por etiqueta** "
    "o **Búsqueda por documento**."
)
