from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from app.config import get_settings


def retrieve_documents(
    vector_store: VectorStore,
    query: str,
    top_k: int | None = None,
) -> list[Document]:
    """Retrieve the most relevant documents for a user query."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    settings = get_settings()

    k = top_k if top_k is not None else settings.top_k

    if k <= 0:
        raise ValueError("top_k must be greater than zero")

    return vector_store.similarity_search(
        query=query,
        k=k,
    )
