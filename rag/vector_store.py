"""ChromaDB-backed vector store for the smart-diagnostics RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Default on-disk location for the Chroma persistent store.
DEFAULT_PERSIST_DIR = Path(
    os.environ.get(
        "CHROMA_PERSIST_DIR",
        Path(__file__).resolve().parent / "chroma_db",
    )
)
DEFAULT_COLLECTION = os.environ.get("CHROMA_COLLECTION", "equipment_manuals")
# DEFAULT_EMBEDDING_MODEL = os.environ.get(
#     "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
# )

DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)


def get_embedding_function(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Return a Chroma-compatible embedding function.

    Uses sentence-transformers locally so no API key is required.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


def get_client(persist_dir: Path | str = DEFAULT_PERSIST_DIR) -> chromadb.api.ClientAPI:
    """Create (or open) a persistent Chroma client."""
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_path),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(
    name: str = DEFAULT_COLLECTION,
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
):
    """Return the named Chroma collection, creating it if necessary."""
    client = get_client(persist_dir)
    return client.get_or_create_collection(
        name=name,
        embedding_function=get_embedding_function(embedding_model),
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    collection,
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[dict] | None = None,
    batch_size: int = 128,
) -> int:
    """Insert documents into the collection in batches. Returns count added."""
    if len(ids) != len(documents):
        raise ValueError("ids and documents must have the same length")
    if metadatas is not None and len(metadatas) != len(documents):
        raise ValueError("metadatas must align with documents")

    total = 0
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.add(
            ids=list(ids[start:end]),
            documents=list(documents[start:end]),
            metadatas=list(metadatas[start:end]) if metadatas is not None else None,
        )
        total += end - start if end <= len(documents) else len(documents) - start
    return total


def query(
    collection,
    text: str | Iterable[str],
    n_results: int = 4,
    where: dict | None = None,
):
    """Run a similarity search and return Chroma's raw response."""
    query_texts = [text] if isinstance(text, str) else list(text)
    return collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where,
    )
