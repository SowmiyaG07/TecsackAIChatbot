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

EMBEDDING_BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "32")
)

# nomic-embed-text is trained with task prefixes. Documents and
# queries must use the matching prefix or retrieval quality drops.
USE_TASK_PREFIXES = os.getenv(
    "EMBEDDING_TASK_PREFIXES",
    "true" if "nomic" in EMBEDDING_MODEL.lower() else "false",
).strip().lower() in {"1", "true", "yes"}

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# Stored with every vector. When the model or prefix setting
# changes, existing vectors no longer match and are re-embedded.
EMBEDDING_FORMAT = (
    EMBEDDING_MODEL
    + "|"
    + ("prefixed" if USE_TASK_PREFIXES else "raw")
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
# Embedding input formatting
# ---------------------------------------------------------

def format_document_for_embedding(text: str) -> str:
    if USE_TASK_PREFIXES:
        return DOCUMENT_PREFIX + text

    return text


def format_query_for_embedding(text: str) -> str:
    if USE_TASK_PREFIXES:
        return QUERY_PREFIX + text

    return text


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

def to_chroma_value(value: Any) -> str | int | float | bool | None:
    """
    Chroma metadata values must be primitive types.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    # heading_path: ["Project", "Architecture"] -> "Project > Architecture"
    if isinstance(value, (list, tuple)):
        items = [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ]

        return " > ".join(items) if items else None

    return str(value)


def get_metadata(
    chunk: dict[str, Any],
) -> dict[str, Any]:

    metadata = chunk.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    metadata = dict(metadata)

    # Preserve useful top-level fields too.
    for field in [
        "document_id",
        "file_name",
        "relative_path",
        "top_level",
        "project",
        "section",
        "subsection",
        "heading_path",
        "page_start",
        "page_end",
        "chunk_index",
        "version",
        "content_type",
        "file_hash",
    ]:
        if field in chunk and field not in metadata:
            metadata[field] = chunk[field]

    metadata["embedding_format"] = EMBEDDING_FORMAT

    clean_metadata = {}

    for key, value in metadata.items():

        value = to_chroma_value(value)

        if value is None:
            continue

        clean_metadata[key] = value

    return clean_metadata


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate document embeddings in a single Ollama request.

    Callers are responsible for batching.
    """

    if not texts:
        return []

    response = ollama_client.embed(
        model=EMBEDDING_MODEL,
        input=[
            format_document_for_embedding(text)
            for text in texts
        ],
    )

    embeddings = response["embeddings"]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch. "
            f"Expected {len(texts)}, "
            f"got {len(embeddings)}."
        )

    return embeddings


# ---------------------------------------------------------
# Sync
# ---------------------------------------------------------

def build_desired_state(
    chunks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    chunk_id -> {"text": ..., "metadata": ...}
    for every valid chunk on disk.
    """

    desired: dict[str, dict[str, Any]] = {}

    for index, chunk in enumerate(chunks):

        text = get_chunk_text(chunk)

        if not text:
            continue

        desired[get_chunk_id(chunk, index)] = {
            "text": text,
            "metadata": get_metadata(chunk),
        }

    return desired


def load_existing_state() -> dict[str, dict[str, Any]]:

    data = collection.get(
        include=["documents", "metadatas"],
    )

    existing: dict[str, dict[str, Any]] = {}

    for index, chunk_id in enumerate(data.get("ids", [])):
        existing[chunk_id] = {
            "text": data["documents"][index],
            "metadata": data["metadatas"][index] or {},
        }

    return existing


def sync_vector_store() -> None:
    """
    Make ChromaDB match storage/chunks exactly.

    - Vectors whose chunk no longer exists are deleted
      (deleted documents, or documents that now produce
      fewer chunks).
    - Chunks that are new, or whose text/metadata/embedding
      format changed, are (re-)embedded.
    - Everything else is left untouched.
    """

    print("=" * 70)
    print("VECTOR STORE SYNC")
    print("=" * 70)
    print(f"Chunks directory : {CHUNKS_DIR}")
    print(f"Embedding model  : {EMBEDDING_MODEL}")
    print(f"Task prefixes    : {USE_TASK_PREFIXES}")
    print(f"Chroma path      : {CHROMA_PATH}")
    print(f"Collection       : {CHROMA_COLLECTION}")

    chunks = load_chunks()

    desired = build_desired_state(chunks)

    existing = load_existing_state()

    print(f"Chunks on disk   : {len(desired)}")
    print(f"Vectors stored   : {len(existing)}")

    # -----------------------------------------------------
    # Delete stale vectors
    # -----------------------------------------------------

    stale_ids = [
        chunk_id
        for chunk_id in existing
        if chunk_id not in desired
    ]

    for start in range(0, len(stale_ids), 500):
        collection.delete(
            ids=stale_ids[start:start + 500],
        )

    print(f"Deleted stale    : {len(stale_ids)}")

    # -----------------------------------------------------
    # Embed new / changed chunks
    # -----------------------------------------------------

    to_embed = [
        chunk_id
        for chunk_id, item in desired.items()
        if chunk_id not in existing
        or existing[chunk_id]["text"] != item["text"]
        or existing[chunk_id]["metadata"] != item["metadata"]
    ]

    print(f"To embed         : {len(to_embed)}")

    dimension = None

    for start in range(0, len(to_embed), EMBEDDING_BATCH_SIZE):

        batch_ids = to_embed[start:start + EMBEDDING_BATCH_SIZE]

        print(
            f"Embedding chunks {start + 1}-"
            f"{start + len(batch_ids)} of {len(to_embed)}..."
        )

        texts = [desired[i]["text"] for i in batch_ids]

        embeddings = generate_embeddings(texts)

        if embeddings:
            dimension = len(embeddings[0])

        collection.upsert(
            ids=batch_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[desired[i]["metadata"] for i in batch_ids],
        )

    print()
    print("=" * 70)
    print("VECTOR STORE SUMMARY")
    print("=" * 70)

    print(f"Chunks stored    : {collection.count()}")

    if dimension:
        print(f"Dimensions       : {dimension}")

    print(f"Collection       : {CHROMA_COLLECTION}")

    print("=" * 70)


# Backwards-compatible name.
build_vector_store = sync_vector_store


if __name__ == "__main__":
    sync_vector_store()
