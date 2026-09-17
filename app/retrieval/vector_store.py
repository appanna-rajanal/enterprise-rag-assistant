from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import get_settings
from app.retrieval.embeddings import create_embeddings


def create_vector_store(documents: list[Document]) -> FAISS:
    """Create an in-memory FAISS vector store from documents."""

    if not documents:
        raise ValueError("Cannot create a vector store from empty documents")

    embeddings = create_embeddings()

    return FAISS.from_documents(
        documents,
        embeddings,
    )


def save_vector_store(vector_store: FAISS) -> None:
    """Persist the FAISS vector store locally."""

    settings = get_settings()
    path = Path(settings.vector_store_path)

    path.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(path))


def load_vector_store() -> FAISS:
    """Load the persisted FAISS vector store."""

    settings = get_settings()
    path = Path(settings.vector_store_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Vector store does not exist: {path}"
        )

    embeddings = create_embeddings()

    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
