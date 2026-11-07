import json

import pytest

from rag_documentos.searching import global_search
from rag_documentos.searching.global_search import SearchResult


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeCollection:
    def __init__(self, results=None, count=0) -> None:
        self.results = results or {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        self.chunk_count = count
        self.query_arguments = None

    def count(self) -> int:
        return self.chunk_count

    def query(self, **kwargs):
        self.query_arguments = kwargs
        return self.results


class FakeOllamaClient:
    def __init__(self, relevant_ids: list[int]) -> None:
        self.relevant_ids = relevant_ids
        self.messages = []

    def chat(self, **kwargs):
        self.messages.append(kwargs["messages"])
        return FakeChatResponse(json.dumps({"relevant_ids": self.relevant_ids}))


def test_search_returns_best_excerpt_per_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = FakeCollection(
        count=6,
        results={
            "documents": [[
                "programación fragmento A",
                "mejor fragmento de programación",
                "información de código",
                "fragmento sobre algoritmos",
                "otro fragmento del mismo documento",
                "último documento relacionado",
            ]],
            "metadatas": [[
                {"source": "a.docx"},
                {"source": "a.docx"},
                {"source": "b.pdf"},
                {"source": "c.xlsx"},
                {"source": "b.pdf"},
                {"source": "d.docx"},
            ]],
            "distances": [[0.4, 0.1, 0.2, 0.3, 0.6, 0.5]],
        },
    )
    ollama_client = FakeOllamaClient([0, 1, 2, 3])
    monkeypatch.setattr(global_search, "get_collection", lambda: collection)
    monkeypatch.setattr(global_search, "_embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(global_search, "_ollama_client", lambda: ollama_client)

    results = global_search.search_documents(
        "¿En qué documento hay información sobre programación?"
    )

    assert results == [
        SearchResult("a.docx", "mejor fragmento de programación", 0.1),
        SearchResult("b.pdf", "información de código", 0.2),
        SearchResult("c.xlsx", "fragmento sobre algoritmos", 0.3),
        SearchResult("d.docx", "último documento relacionado", 0.5),
    ]
    assert collection.query_arguments["n_results"] == 6
    assert collection.query_arguments["include"] == [
        "documents",
        "metadatas",
        "distances",
    ]
    assert len(ollama_client.messages) == 1


def test_search_limits_results_to_five(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = [f"document-{index}.pdf" for index in range(7)]
    collection = FakeCollection(
        count=len(sources),
        results={
            "documents": [[f"fragmento {index}" for index in range(len(sources))]],
            "metadatas": [[{"source": source} for source in sources]],
            "distances": [[float(index) for index in range(len(sources))]],
        },
    )
    monkeypatch.setattr(
        global_search, "_filter_relevant_results", lambda query, results: results
    )
    monkeypatch.setattr(global_search, "get_collection", lambda: collection)
    monkeypatch.setattr(global_search, "_embed_texts", lambda texts: [[0.1]])

    results = global_search.search_documents("programación", max_results=5)

    assert len(results) == 5
    assert results[-1].source == "document-4.pdf"


def test_search_excludes_documents_not_confirmed_relevant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = FakeCollection(
        count=3,
        results={
            "documents": [[
                "Curriculum vitae de una desarrolladora con experiencia en Python.",
                "Manual de instalación de una impresora.",
                "Receta para preparar pan casero.",
            ]],
            "metadatas": [[
                {"source": "cv.docx"},
                {"source": "impresora.pdf"},
                {"source": "receta.pdf"},
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        },
    )
    ollama_client = FakeOllamaClient([0])
    monkeypatch.setattr(global_search, "get_collection", lambda: collection)
    monkeypatch.setattr(global_search, "_embed_texts", lambda texts: [[0.1]])
    monkeypatch.setattr(global_search, "_ollama_client", lambda: ollama_client)

    results = global_search.search_documents("Dame los CV que tengas")

    assert results == [
        SearchResult(
            "cv.docx",
            "Curriculum vitae de una desarrolladora con experiencia en Python.",
            0.1,
        )
    ]


def test_search_returns_empty_when_collection_has_no_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = FakeCollection()
    monkeypatch.setattr(global_search, "get_collection", lambda: collection)
    monkeypatch.setattr(
        global_search,
        "_embed_texts",
        lambda texts: pytest.fail("No debe crear embedding para un índice vacío"),
    )
    monkeypatch.setattr(
        global_search,
        "_ollama_client",
        lambda: pytest.fail("No debe llamar al modelo para un índice vacío"),
    )

    assert global_search.search_documents("programación") == []


@pytest.mark.parametrize(
    ("query", "max_results"),
    [("", 5), ("   ", 5), ("programación", 0), ("programación", 6)],
)
def test_search_rejects_empty_query_and_invalid_result_limits(
    query: str, max_results: int
) -> None:
    with pytest.raises(ValueError):
        global_search.search_documents(query, max_results)
