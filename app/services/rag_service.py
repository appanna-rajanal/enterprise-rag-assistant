from langchain_core.documents import Document

from app.config import get_settings
from app.generation.bedrock import generate_answer
from app.ingestion.chunker import split_documents
from app.ingestion.loader import load_documents
from app.retrieval.retriever import retrieve_documents
from app.retrieval.vector_store import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
)


class RAGService:
    """Orchestrates document ingestion, retrieval and generation."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_store = None

    def build_index(self) -> int:
        """Load, chunk and index documents."""

        documents = load_documents(
            self.settings.documents_path
        )

        chunks = split_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        self.vector_store = create_vector_store(chunks)
        save_vector_store(self.vector_store)

        return len(chunks)

    def load_index(self) -> None:
        """Load the persisted vector store."""

        self.vector_store = load_vector_store()

    def query(self, question: str) -> dict:
        """Retrieve context and generate a grounded answer."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if self.vector_store is None:
            self.load_index()

        documents: list[Document] = retrieve_documents(
            self.vector_store,
            question,
            top_k=self.settings.top_k,
        )

        context = self._build_context(documents)

        answer = generate_answer(
            question=question,
            context=context,
        )

        sources = [
            document.metadata.get(
                "source",
                "unknown",
            )
            for document in documents
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_documents": len(documents),
        }

    @staticmethod
    def _build_context(
        documents: list[Document],
    ) -> str:
        """Combine retrieved documents into LLM context."""

        return "\n\n".join(
            document.page_content
            for document in documents
        )
