from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import chromadb
from ollama import Client



load_dotenv()
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_DIR = PROJECT_ROOT / "storage" / "chunks"

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    str(PROJECT_ROOT / "chroma_db"),
)

CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "company_documents",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text",
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


# ---------------------------------------------------------
# Clients
# ---------------------------------------------------------

ollama_client = Client(
    host=OLLAMA_BASE_URL
)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={
        "hnsw:space": "cosine"
    },
)


# ---------------------------------------------------------
# Load chunks
# ---------------------------------------------------------

def load_chunks() -> list[dict[str, Any]]:
    """
    Load all chunk JSONL files from storage/chunks.
    Each line contains one chunk JSON object.
    """

    chunk_files = sorted(CHUNKS_DIR.glob("*.jsonl"))

    if not chunk_files:
        raise FileNotFoundError(
            f"No .jsonl chunk files found in: {CHUNKS_DIR}"
        )

    chunks: list[dict[str, Any]] = []

    for file_path in chunk_files:
        with file_path.open("r", encoding="utf-8") as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                chunk = json.loads(line)

                if isinstance(chunk, dict):
                    chunks.append(chunk)

    return chunks


# ---------------------------------------------------------
# Text extraction
# ---------------------------------------------------------

def get_chunk_text(chunk: dict[str, Any]) -> str:
    """
    Extract text from a chunk.
    """

    text = (
        chunk.get("text")
        or chunk.get("content")
        or chunk.get("page_content")
        or ""
    )

    return str(text).strip()


# ---------------------------------------------------------
# Chunk ID
# ---------------------------------------------------------

def get_chunk_id(
    chunk: dict[str, Any],
    index: int,
) -> str:
    """
    Generate a stable chunk ID.
    """

    return str(
        chunk.get("chunk_id")
        or chunk.get("id")
        or f"chunk_{index:06d}"
    )


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

def get_metadata(
    chunk: dict[str, Any],
) -> dict[str, Any]:

    metadata = chunk.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    # Preserve useful top-level fields too.
    for field in [
        "document_id",
        "file_name",
        "relative_path",
        "top_level",
        "project",
        "section",
        "subsection",
        "page",
        "chunk_index",
        "version",
        "content_type",
        "file_hash",
    ]:
        if field in chunk and field not in metadata:
            metadata[field] = chunk[field]

    # Chroma metadata values must be primitive types.
    clean_metadata = {}

    for key, value in metadata.items():

        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            clean_metadata[key] = value

        else:
            clean_metadata[key] = str(value)

    return clean_metadata


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

def generate_embeddings(
    texts: list[str],
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Generate embeddings in batches to avoid oversized Ollama requests.
    """

    all_embeddings: list[list[float]] = []

    total = len(texts)

    print(
        f"Generating embeddings for {total} chunks..."
    )

    for start in range(0, total, batch_size):

        end = min(
            start + batch_size,
            total,
        )

        batch = texts[start:end]

        print(
            f"Embedding chunks {start + 1}-{end} "
            f"of {total}..."
        )

        response = ollama_client.embed(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        embeddings = response["embeddings"]

        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch. "
                f"Expected {len(batch)}, "
                f"got {len(embeddings)}."
            )

        all_embeddings.extend(embeddings)

    print(
        f"Generated {len(all_embeddings)} embeddings"
    )

    if all_embeddings:
        print(
            f"Embedding dimension: "
            f"{len(all_embeddings[0])}"
        )

    return all_embeddings


def store_embeddings(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> None:

    ids = []
    documents = []
    metadatas = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        text = get_chunk_text(chunk)

        if not text:
            continue

        ids.append(
            get_chunk_id(chunk, index)
        )

        documents.append(text)

        metadatas.append(
            get_metadata(chunk)
        )

    if not ids:
        raise ValueError(
            "No valid chunks available for ChromaDB."
        )

    print(
        f"Adding {len(ids)} chunks to ChromaDB..."
    )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("Embeddings stored successfully.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def build_vector_store() -> None:

    print("=" * 70)
    print("EMBEDDING GENERATION")
    print("=" * 70)

    print(f"Chunks directory : {CHUNKS_DIR}")
    print(f"Embedding model  : {EMBEDDING_MODEL}")
    print(f"Chroma path      : {CHROMA_PATH}")
    print(f"Collection       : {CHROMA_COLLECTION}")

    chunks = load_chunks()

    print(
        f"Chunks loaded    : {len(chunks)}"
    )

    texts = []

    valid_chunks = []

    for chunk in chunks:

        text = get_chunk_text(chunk)

        if not text:
            continue

        texts.append(text)
        valid_chunks.append(chunk)

    print(
        f"Valid chunks     : {len(valid_chunks)}"
    )

    embeddings = generate_embeddings(texts)

    if len(embeddings) != len(valid_chunks):
        raise RuntimeError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    store_embeddings(
        valid_chunks,
        embeddings,
    )

    print()
    print("=" * 70)
    print("VECTOR STORE SUMMARY")
    print("=" * 70)

    print(
        f"Chunks stored    : {collection.count()}"
    )

    if embeddings:
        print(
            f"Dimensions       : {len(embeddings[0])}"
        )

    print(
        f"Collection       : {CHROMA_COLLECTION}"
    )

    print("=" * 70)


if __name__ == "__main__":
    build_vector_store()
