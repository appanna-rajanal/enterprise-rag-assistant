from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents(documents_path: str) -> list[Document]:
    """Load supported documents from a directory."""

    directory = Path(documents_path)

    if not directory.exists():
        raise FileNotFoundError(
            f"Documents directory does not exist: {documents_path}"
        )

    documents: list[Document] = []

    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

        documents.extend(loader.load())

    if not documents:
        raise ValueError(
            f"No supported documents found in: {documents_path}"
        )

    return documents
