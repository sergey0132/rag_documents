import json

import pytest

from rag_documentos.tagging import document_tags
from rag_documentos.tagging.document_tags import TaggedDocument


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeOllamaClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return FakeChatResponse(json.dumps(self.response))


def test_classify_document_uses_fixed_category_and_subcategory_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeOllamaClient({"category": "CV", "subcategory": "Perfil técnico"})
    monkeypatch.setattr(document_tags, "_ollama_client", lambda: client)

    classification = document_tags.classify_document(
        "Currículum vitae y experiencia."
    )

    assert classification == document_tags.DocumentClassification(
        category="CV",
        subcategory="Perfil técnico",
    )
    assert client.calls[0]["format"] == document_tags.CLASSIFICATION_RESPONSE_FORMAT
    assert client.calls[0]["format"]["properties"]["category"]["enum"] == list(
        document_tags.CATEGORIES
    )
    assert "Currículum vitae" in client.calls[0]["messages"][1]["content"]


def test_classify_document_returns_book_and_genre(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeOllamaClient({"category": "Libro", "subcategory": "Novela"})
    monkeypatch.setattr(document_tags, "_ollama_client", lambda: client)

    classification = document_tags.classify_document("Texto de una novela.")

    assert classification == document_tags.DocumentClassification("Libro", "Novela")


def test_classify_document_rejects_empty_or_invalid_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="sin texto"):
        document_tags.classify_document("  ")

    monkeypatch.setattr(
        document_tags,
        "_ollama_client",
        lambda: FakeOllamaClient(
            {"category": "Documento", "subcategory": "General"}
        ),
    )
    with pytest.raises(ValueError, match="categoría no permitida"):
        document_tags.classify_document("Contenido del documento")

    monkeypatch.setattr(
        document_tags,
        "_ollama_client",
        lambda: FakeOllamaClient({"category": "Libro", "subcategory": "x" * 51}),
    )
    with pytest.raises(ValueError, match="no puede superar"):
        document_tags.classify_document("Contenido del documento")


def test_normalize_custom_tag_is_optional_and_validated() -> None:
    assert document_tags.normalize_custom_tag("  Partitura  ") == "Partitura"
    assert document_tags.normalize_custom_tag("   ") == "Otro"

    with pytest.raises(ValueError, match="no puede superar"):
        document_tags.normalize_custom_tag("x" * 51)


def test_get_document_tags_returns_unique_current_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def get(self, where, include):
            assert where == {"source": "libro.pdf"}
            assert include == ["metadatas"]
            return {
                "metadatas": [
                    {"source": "libro.pdf", "tag": "Libro"},
                    {"source": "libro.pdf", "tag": "Libro"},
                    {"source": "libro.pdf", "tag": "Novela"},
                ]
            }

    monkeypatch.setattr(document_tags, "get_collection", FakeCollection)

    assert document_tags.get_document_tags("libro.pdf") == ["Libro", "Novela"]


def test_update_document_tag_changes_all_chunks_and_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def __init__(self) -> None:
            self.updated_ids = []
            self.updated_metadatas = []

        def get(self, where, include):
            assert where == {"source": "libro.pdf"}
            assert include == ["metadatas"]
            return {
                "ids": ["chunk-1", "chunk-2"],
                "metadatas": [
                    {
                        "source": "libro.pdf",
                        "tag": "Manual",
                        "category": "Libro",
                        "subcategory": "Novela",
                    },
                    {
                        "source": "libro.pdf",
                        "tag": "Manual",
                        "category": "Libro",
                        "subcategory": "Novela",
                    },
                ],
            }

        def update(self, ids, metadatas):
            self.updated_ids = ids
            self.updated_metadatas = metadatas

    collection = FakeCollection()
    monkeypatch.setattr(document_tags, "get_collection", lambda: collection)

    updated_count = document_tags.update_document_tag("libro.pdf", "  Literatura  ")

    assert updated_count == 2
    assert collection.updated_ids == ["chunk-1", "chunk-2"]
    assert all(
        metadata["tag"] == "Literatura"
        and metadata["category"] == "Libro"
        and metadata["subcategory"] == "Novela"
        for metadata in collection.updated_metadatas
    )


@pytest.mark.parametrize(
    ("document_name", "new_tag", "error"),
    [
        ("libro.pdf", " ", "no puede estar vacía"),
        ("libro.pdf", "x" * 51, "no puede superar"),
        ("ausente.pdf", "Libro", "no está indexado"),
        (" ", "Libro", "Selecciona un documento"),
    ],
)
def test_update_document_tag_rejects_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
    document_name: str,
    new_tag: str,
    error: str,
) -> None:
    class FakeCollection:
        def get(self, where, include):
            return {"ids": [], "metadatas": []}

    monkeypatch.setattr(document_tags, "get_collection", FakeCollection)

    with pytest.raises(ValueError, match=error):
        document_tags.update_document_tag(document_name, new_tag)


def test_tag_unlabeled_documents_updates_only_missing_document_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def __init__(self) -> None:
            self.updated_ids = []
            self.updated_metadatas = []

        def get(self, include):
            assert include == ["documents", "metadatas"]
            return {
                "ids": ["cv-0", "cv-1", "manual-0"],
                "documents": ["CV primera parte", "CV segunda parte", "Manual"],
                "metadatas": [
                    {"source": "cv.pdf", "chunk_index": 0},
                    {"source": "cv.pdf", "chunk_index": 1},
                    {"source": "manual.pdf", "tag": "Manual", "chunk_index": 0},
                ],
            }

        def update(self, ids, metadatas):
            self.updated_ids = ids
            self.updated_metadatas = metadatas

    collection = FakeCollection()
    monkeypatch.setattr(document_tags, "get_collection", lambda: collection)
    monkeypatch.setattr(
        document_tags,
        "classify_document",
        lambda text: document_tags.DocumentClassification("CV", "Perfil técnico"),
    )

    updated = document_tags.tag_unlabeled_documents()

    assert updated == {"cv.pdf": "CV"}
    assert collection.updated_ids == ["cv-0", "cv-1"]
    assert collection.updated_metadatas == [
        {"source": "cv.pdf", "chunk_index": 0, "tag": "CV"},
        {"source": "cv.pdf", "chunk_index": 1, "tag": "CV"},
    ]


def test_list_untagged_documents_finds_documents_missing_any_chunk_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def get(self, include):
            assert include == ["metadatas"]
            return {
                "metadatas": [
                    {"source": "manual.pdf", "tag": "Manual"},
                    {"source": "old-cv.pdf"},
                    {"source": "old-cv.pdf", "tag": "CV"},
                    {"source": "invoice.pdf", "tag": "Factura"},
                ]
            }

    monkeypatch.setattr(document_tags, "get_collection", FakeCollection)

    assert document_tags.list_untagged_documents() == ["old-cv.pdf"]


def test_list_existing_tags_returns_unique_sorted_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def get(self, include):
            assert include == ["metadatas"]
            return {
                "metadatas": [
                    {"source": "cv.pdf", "tag": "CV"},
                    {"source": "cv.pdf", "tag": "CV"},
                    {"source": "manual.pdf", "tag": "Manual"},
                    {"source": "old.pdf"},
                    None,
                ]
            }

    monkeypatch.setattr(document_tags, "get_collection", FakeCollection)

    assert document_tags.list_existing_tags() == ["CV", "Manual"]


def test_search_documents_by_tag_maps_free_text_and_returns_all_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def __init__(self) -> None:
            self.filter = None

        def get(self, include, where=None):
            if where is None:
                assert include == ["metadatas"]
                return {
                    "metadatas": [
                        {"source": "cv-one.pdf", "tag": "CV", "chunk_index": 0},
                        {"source": "cv-two.docx", "tag": "CV", "chunk_index": 0},
                        {"source": "manual.pdf", "tag": "Manual", "chunk_index": 0},
                    ]
                }

            self.filter = where
            assert include == ["documents", "metadatas"]
            return {
                "documents": [
                    "CV primera parte",
                    "CV segunda parte",
                    "Segundo currículum",
                ],
                "metadatas": [
                    {"source": "cv-one.pdf", "tag": "CV", "chunk_index": 0},
                    {"source": "cv-one.pdf", "tag": "CV", "chunk_index": 1},
                    {"source": "cv-two.docx", "tag": "CV", "chunk_index": 0},
                ],
            }

    collection = FakeCollection()
    client = FakeOllamaClient({"tag": "cv"})
    monkeypatch.setattr(document_tags, "get_collection", lambda: collection)
    monkeypatch.setattr(document_tags, "_ollama_client", lambda: client)

    results = document_tags.search_documents_by_tag(
        "Necesito los currículums de las personas candidatas"
    )

    assert collection.filter == {"tag": "CV"}
    assert results == [
        TaggedDocument(
            name="cv-one.pdf",
            tag="CV",
            excerpt="CV primera parte\n\nCV segunda parte",
        ),
        TaggedDocument(
            name="cv-two.docx",
            tag="CV",
            excerpt="Segundo currículum",
        ),
    ]


def test_search_documents_by_tag_returns_empty_without_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def get(self, include):
            return {"metadatas": [{"source": "old.pdf"}]}

    monkeypatch.setattr(document_tags, "get_collection", FakeCollection)
    monkeypatch.setattr(
        document_tags,
        "_ollama_client",
        lambda: pytest.fail("No debe llamar al LLM si no hay etiquetas"),
    )

    assert document_tags.search_documents_by_tag("Busca los CV") == []


def test_search_documents_by_tag_rejects_unknown_llm_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def get(self, include, where=None):
            if where is None:
                return {"metadatas": [{"source": "manual.pdf", "tag": "Manual"}]}
            pytest.fail("No se debe consultar una etiqueta desconocida")

    monkeypatch.setattr(document_tags, "get_collection", FakeCollection)
    monkeypatch.setattr(
        document_tags,
        "_ollama_client",
        lambda: FakeOllamaClient({"tag": "CV"}),
    )

    with pytest.raises(ValueError, match="no existe"):
        document_tags.search_documents_by_tag("Busca los CV")


def test_search_documents_by_tag_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="Escribe"):
        document_tags.search_documents_by_tag(" ")
