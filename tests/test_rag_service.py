from io import BytesIO

import pytest
from docx import Document

from rag_documentos import rag_service
from rag_documentos.rag_service import RetrievedChunk
from rag_documentos.tagging.document_tags import DocumentClassification


def test_embed_texts_sends_chunks_in_bounded_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEmbeddingResponse:
        def __init__(self, embeddings) -> None:
            self.embeddings = embeddings

    class FakeOllamaClient:
        def __init__(self) -> None:
            self.batch_sizes = []

        def embed(self, model, input):
            self.batch_sizes.append(len(input))
            return FakeEmbeddingResponse([[float(index)] for index, _ in enumerate(input)])

    client = FakeOllamaClient()
    monkeypatch.setattr(rag_service, "_ollama_client", lambda: client)

    embeddings = rag_service._embed_texts(
        ["fragmento"] * (rag_service.EMBEDDING_BATCH_SIZE + 3)
    )

    assert client.batch_sizes == [rag_service.EMBEDDING_BATCH_SIZE, 3]
    assert len(embeddings) == rag_service.EMBEDDING_BATCH_SIZE + 3


def test_embed_texts_rejects_incomplete_ollama_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEmbeddingResponse:
        embeddings = []

    class FakeOllamaClient:
        def embed(self, model, input):
            return FakeEmbeddingResponse()

    monkeypatch.setattr(rag_service, "_ollama_client", FakeOllamaClient)

    with pytest.raises(RuntimeError, match="cantidad de embeddings"):
        rag_service._embed_texts(["fragmento"])


def test_split_text_uses_500_character_chunks_and_50_overlap() -> None:
    text = "a" * 1_000

    chunks = rag_service.split_text(text)

    assert [len(chunk) for chunk in chunks] == [500, 500, 100]
    assert chunks[0][-50:] == chunks[1][:50]
    assert chunks[1][-50:] == chunks[2][:50]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_split_text_rejects_invalid_chunk_settings(
    chunk_size: int, chunk_overlap: int
) -> None:
    with pytest.raises(ValueError):
        rag_service.split_text("texto", chunk_size, chunk_overlap)


def test_index_document_stores_custom_tag_for_other_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def __init__(self) -> None:
            self.ids = []
            self.metadatas = []

        def upsert(self, ids, documents, embeddings, metadatas):
            self.ids = ids
            self.metadatas = metadatas

        def get(self, where, include):
            return {"ids": self.ids}

    collection = FakeCollection()
    monkeypatch.setattr(rag_service, "get_collection", lambda: collection)
    monkeypatch.setattr(rag_service, "load_document", lambda filename, file: "texto")
    monkeypatch.setattr(rag_service, "split_text", lambda text: ["fragmento"])
    monkeypatch.setattr(rag_service, "_embed_texts", lambda chunks: [[0.1]])

    count = rag_service.index_document(
        "partitura.pdf",
        b"pdf",
        classification=DocumentClassification("Otro", "Música"),
        custom_tag="  Partitura  ",
    )

    assert count == 1
    assert collection.metadatas[0]["source"] == "partitura.pdf"
    assert collection.metadatas[0]["chunk_index"] == 0
    assert collection.metadatas[0]["tag"] == "Partitura"
    assert collection.metadatas[0]["category"] == "Otro"
    assert collection.metadatas[0]["subcategory"] == "Música"


def test_index_document_rejects_custom_tag_for_fixed_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag_service, "load_document", lambda filename, file: "texto")
    monkeypatch.setattr(rag_service, "split_text", lambda text: ["fragmento"])

    with pytest.raises(ValueError, match="solo se admite"):
        rag_service.index_document(
            "informe.pdf",
            b"pdf",
            classification=DocumentClassification("Informe", "Ventas"),
            custom_tag="Personalizada",
        )


def test_index_document_stores_chunks_and_replaces_old_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def __init__(self) -> None:
            self.items: dict[str, dict[str, object]] = {}

        def upsert(self, ids, documents, embeddings, metadatas) -> None:
            for item_id, document, embedding, metadata in zip(
                ids, documents, embeddings, metadatas
            ):
                self.items[item_id] = {
                    "document": document,
                    "embedding": embedding,
                    "metadata": metadata,
                }

        def get(self, where, include):
            matching_ids = [
                item_id
                for item_id, item in self.items.items()
                if item["metadata"]["source"] == where["source"]
            ]
            return {"ids": matching_ids}

        def delete(self, ids) -> None:
            for item_id in ids:
                self.items.pop(item_id)

    document = Document()
    document.add_paragraph("contenido anterior " * 40)
    old_buffer = BytesIO()
    document.save(old_buffer)
    old_content = old_buffer.getvalue()

    document = Document()
    document.add_paragraph("contenido actualizado " * 40)
    new_buffer = BytesIO()
    document.save(new_buffer)
    new_content = new_buffer.getvalue()

    collection = FakeCollection()
    monkeypatch.setattr(rag_service, "get_collection", lambda: collection)
    monkeypatch.setattr(
        rag_service, "_embed_texts", lambda chunks: [[0.1] for _ in chunks]
    )
    monkeypatch.setattr(
        "rag_documentos.tagging.document_tags.classify_document",
        lambda text: DocumentClassification("Informe", "Ventas"),
    )

    assert rag_service.index_document("informe.docx", old_content) > 0
    first_ids = set(collection.items)
    assert rag_service.index_document("informe.docx", new_content) > 0

    assert set(collection.items).isdisjoint(first_ids)
    assert all(
        item["metadata"]["source"] == "informe.docx"
        for item in collection.items.values()
    )
    assert all(
        item["metadata"]["tag"] == "Informe"
        and item["metadata"]["category"] == "Informe"
        and item["metadata"]["subcategory"] == "Ventas"
        for item in collection.items.values()
    )


def test_ask_question_returns_answer_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCollection:
        def query(self, query_embeddings, n_results, include):
            assert n_results == 4
            return {
                "documents": [["El horario es de nueve a cinco."]],
                "metadatas": [[{"source": "horario.docx"}]],
            }

    class FakeMessage:
        content = "El horario es de nueve a cinco. [horario.docx]"

    class FakeResponse:
        message = FakeMessage()

    class FakeOllamaClient:
        def chat(self, model, messages):
            assert "horario.docx" in messages[1]["content"]
            return FakeResponse()

    monkeypatch.setattr(rag_service, "get_collection", FakeCollection)
    monkeypatch.setattr(rag_service, "_embed_texts", lambda texts: [[0.1]])
    monkeypatch.setattr(rag_service, "_ollama_client", FakeOllamaClient)

    result = rag_service.ask_question("¿Cuál es el horario?")

    assert result.answer == "El horario es de nueve a cinco. [horario.docx]"
    assert result.sources == [
        RetrievedChunk(source="horario.docx", text="El horario es de nueve a cinco.")
    ]


def test_list_indexed_documents_returns_unique_names_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def get(self, include):
            assert include == ["metadatas"]
            return {
                "metadatas": [
                    {"source": "zeta.pdf"},
                    {"source": "informe.docx"},
                    {"source": "zeta.pdf"},
                    {"source": ""},
                    None,
                ]
            }

    monkeypatch.setattr(rag_service, "get_collection", FakeCollection)

    assert rag_service.list_indexed_documents() == ["informe.docx", "zeta.pdf"]


def test_ask_document_question_restricts_retrieval_to_selected_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def get(self, where, include):
            assert where == {"source": "cv.docx"}
            assert include == ["metadatas"]
            return {"ids": ["cv-chunk-1", "cv-chunk-2"]}

        def query(self, **kwargs):
            assert kwargs["where"] == {"source": "cv.docx"}
            assert kwargs["n_results"] == 2
            return {
                "documents": [["Experiencia en desarrollo de software."]],
                "metadatas": [[{"source": "cv.docx"}]],
            }

    class FakeMessage:
        content = "El CV menciona experiencia en desarrollo de software."

    class FakeResponse:
        message = FakeMessage()

    class FakeOllamaClient:
        def chat(self, model, messages):
            assert "exclusivamente" in messages[0]["content"]
            assert "[cv.docx]" in messages[1]["content"]
            assert "cv.docx" in messages[0]["content"]
            return FakeResponse()

    monkeypatch.setattr(rag_service, "get_collection", FakeCollection)
    monkeypatch.setattr(rag_service, "_embed_texts", lambda texts: [[0.1]])
    monkeypatch.setattr(rag_service, "_ollama_client", FakeOllamaClient)

    result = rag_service.ask_document_question(
        "cv.docx", "¿Qué experiencia tiene el candidato?", top_k=4
    )

    assert result.answer == "El CV menciona experiencia en desarrollo de software."
    assert result.sources == [
        RetrievedChunk(
            source="cv.docx",
            text="Experiencia en desarrollo de software.",
        )
    ]


def test_ask_document_question_rejects_unindexed_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def get(self, where, include):
            return {"ids": []}

    monkeypatch.setattr(rag_service, "get_collection", FakeCollection)

    with pytest.raises(ValueError, match="no está indexado"):
        rag_service.ask_document_question("ausente.pdf", "¿De qué trata?")
