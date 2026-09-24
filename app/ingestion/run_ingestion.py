from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ingestion.document_registry import (
    DATA_DIR,
    STORAGE_DIR,
    load_registry,
    save_registry,
    scan_documents,
)

from app.ingestion.document_parser import (
    parse_document,
    DocumentElement,
    print_elements,
)

from app.ingestion.chunker import (
    create_chunks,
    DocumentChunk,
    print_chunks,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CHUNKS_DIR = STORAGE_DIR / "chunks"

CHUNKS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------

def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

def save_json(
    path: Path,
    data: Any,
) -> None:

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temp_path.replace(path)


def save_chunks(
    document_id: str,
    chunks: list[DocumentChunk],
) -> Path:

    output_path = (
        CHUNKS_DIR
        / f"{document_id}.jsonl"
    )

    temp_path = output_path.with_suffix(
        ".jsonl.tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for chunk in chunks:

            json.dump(
                chunk.to_dict(),
                file,
                ensure_ascii=False,
            )

            file.write("\n")

    temp_path.replace(output_path)

    return output_path


# ---------------------------------------------------------------------
# Save parsed elements
# ---------------------------------------------------------------------

def save_elements(
    document_id: str,
    elements: list[DocumentElement],
) -> Path:

    output_path = (
        CHUNKS_DIR
        / f"{document_id}_elements.json"
    )

    save_json(
        output_path,
        [
            element.to_dict()
            for element in elements
        ],
    )

    return output_path


# ---------------------------------------------------------------------
# Registry update
# ---------------------------------------------------------------------

def mark_processed(
    document_id: str,
) -> None:

    registry = load_registry()

    documents = registry.get(
        "documents",
        {},
    )

    record = documents.get(document_id)

    if record is None:
        return

    record["last_processed_at"] = utc_now()
    record["status"] = "processed"

    # Clear errors from previous failed attempts
    record.pop("error_message", None)
    record.pop("last_failed_at", None)

    save_registry(registry)


def mark_failed(
    document_id: str,
    error: Exception,
) -> None:

    registry = load_registry()

    documents = registry.get(
        "documents",
        {},
    )

    record = documents.get(document_id)

    if record is None:
        return

    record["status"] = "failed"

    record["error_message"] = str(error)

    record["last_failed_at"] = utc_now()

    save_registry(registry)


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def build_chunk_metadata(
    document_record: dict[str, Any],
) -> dict[str, Any]:

    return {
        "document_id": document_record[
            "document_id"
        ],

        "file_name": document_record[
            "file_name"
        ],

        "relative_path": document_record[
            "relative_path"
        ],

        "file_extension": document_record[
            "file_extension"
        ],

        "top_level": document_record.get(
            "top_level"
        ),

        "project": document_record.get(
            "project"
        ),

        "document_version": document_record.get(
            "version"
        ),

        "file_hash": document_record.get(
            "file_hash"
        ),
    }


# ---------------------------------------------------------------------
# Process one document
# ---------------------------------------------------------------------

def process_document(
    document_record: dict[str, Any],
    debug: bool = False,
) -> int:

    document_id = document_record[
        "document_id"
    ]

    relative_path = document_record[
        "relative_path"
    ]

    file_path = DATA_DIR / relative_path

    print()
    print("=" * 80)
    print(
        f"PROCESSING: {relative_path}"
    )
    print("=" * 80)

    try:

        # -------------------------------------------------------------
        # Parse
        # -------------------------------------------------------------

        elements = parse_document(
            file_path
        )

        print(
            f"Parsed elements: {len(elements)}"
        )

        if debug:
            print_elements(elements)

        # -------------------------------------------------------------
        # Chunk
        # -------------------------------------------------------------

        base_metadata = build_chunk_metadata(
            document_record
        )

        chunks = create_chunks(
            document_id=document_id,
            elements=elements,
            base_metadata=base_metadata,
        )

        print(
            f"Generated chunks: {len(chunks)}"
        )

        # -------------------------------------------------------------
        # Save intermediate parsed elements
        # -------------------------------------------------------------

        elements_path = save_elements(
            document_id,
            elements,
        )

        print(
            f"Elements saved: {elements_path}"
        )

        # -------------------------------------------------------------
        # Save chunks
        # -------------------------------------------------------------

        chunks_path = save_chunks(
            document_id,
            chunks,
        )

        print(
            f"Chunks saved: {chunks_path}"
        )

        # -------------------------------------------------------------
        # Debug chunks
        # -------------------------------------------------------------

        if debug:
            print_chunks(chunks)

        # -------------------------------------------------------------
        # Mark successfully processed
        # -------------------------------------------------------------

        mark_processed(
            document_id
        )

        print(
            f"SUCCESS: {document_id}"
        )

        return len(chunks)

    except Exception as exc:

        mark_failed(
            document_id,
            exc,
        )

        print()
        print(
            f"FAILED: {document_id}"
        )

        print(
            f"Error: {exc}"
        )

        return 0


# ---------------------------------------------------------------------
# Remove chunks belonging to deleted documents
# ---------------------------------------------------------------------

def cleanup_deleted_documents(
    deleted_documents: list[dict[str, Any]],
) -> None:

    for record in deleted_documents:

        document_id = record[
            "document_id"
        ]

        chunks_path = (
            CHUNKS_DIR
            / f"{document_id}.jsonl"
        )

        elements_path = (
            CHUNKS_DIR
            / f"{document_id}_elements.json"
        )

        if chunks_path.exists():
            chunks_path.unlink()

            print(
                f"Deleted chunks: "
                f"{chunks_path}"
            )

        if elements_path.exists():
            elements_path.unlink()

            print(
                f"Deleted elements: "
                f"{elements_path}"
            )


# ---------------------------------------------------------------------
# Vector store sync
# ---------------------------------------------------------------------

def sync_embeddings() -> None:
    """
    Push chunk changes (new, changed, deleted) into ChromaDB.

    Imported lazily so parsing/chunking works without the
    Ollama/Chroma stack being available.
    """

    from app.embeddings.embedding_service import (
        sync_vector_store,
    )

    print()
    sync_vector_store()


# ---------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------

def run_ingestion(
    debug: bool = False,
    embed: bool = True,
) -> None:

    print()
    print("=" * 80)
    print("TECSACK DOCUMENT INGESTION")
    print("=" * 80)

    # -------------------------------------------------------------
    # Step 1 - scan documents
    # -------------------------------------------------------------

    changes = scan_documents()

    new_documents = changes.get(
        "new",
        [],
    )

    changed_documents = changes.get(
        "changed",
        [],
    )

    deleted_documents = changes.get(
        "deleted",
        [],
    )

    print()
    print("DOCUMENT CHANGES")
    print("-" * 80)

    print(
        f"New:      {len(new_documents)}"
    )

    print(
        f"Changed:  {len(changed_documents)}"
    )

    print(
        f"Deleted:  {len(deleted_documents)}"
    )

    # -------------------------------------------------------------
    # Deleted documents
    # -------------------------------------------------------------

    cleanup_deleted_documents(
        deleted_documents
    )

    # -------------------------------------------------------------
    # Process new documents
    # -------------------------------------------------------------

    documents_to_process = (
        new_documents
        + changed_documents
    )

    if not documents_to_process:

        print()
        print(
            "No new or changed documents."
        )

        # Deleted documents still need their vectors removed.
        if embed and deleted_documents:
            sync_embeddings()

        return

    # -------------------------------------------------------------
    # Process
    # -------------------------------------------------------------

    total_chunks = 0

    successful = 0
    failed = 0

    for record in documents_to_process:

        chunks_count = process_document(
            record,
            debug=debug,
        )

        if chunks_count > 0:

            successful += 1
            total_chunks += chunks_count

        else:

            failed += 1

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("INGESTION SUMMARY")
    print("=" * 80)

    print(
        f"Documents processed: "
        f"{len(documents_to_process)}"
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"Total chunks: {total_chunks}"
    )

    print(
        f"Chunks directory: "
        f"{CHUNKS_DIR}"
    )

    print("=" * 80)

    if embed:
        sync_embeddings()

# ---------------------------------------------------------------------
# Full rebuild
# ---------------------------------------------------------------------


def run_full_rebuild(
    debug: bool = False,
    embed: bool = True,
) -> None:

    print()
    print("=" * 80)
    print("TECSACK FULL DOCUMENT REBUILD")
    print("=" * 80)

    # Refresh the registry first so records carry the current
    # location metadata and deleted files are detected.
    changes = scan_documents()

    cleanup_deleted_documents(
        changes.get("deleted", [])
    )

    registry = load_registry()

    documents = registry.get(
        "documents",
        {},
    )

    print(
        f"Documents found in registry: {len(documents)}"
    )

    total_chunks = 0
    successful = 0
    failed = 0

    for document_id, record in documents.items():

        # Skip deleted documents
        if record.get("status") == "deleted":
            continue

        chunks_count = process_document(
            record,
            debug=debug,
        )

        if chunks_count > 0:

            successful += 1
            total_chunks += chunks_count

        else:

            failed += 1

            print(
                f"WARNING: ZERO CHUNKS: "
                f"{document_id} | "
                f"{record.get('file_name')}"
            )

    print()
    print("=" * 80)
    print("FULL REBUILD SUMMARY")
    print("=" * 80)

    print(
        f"Documents processed: "
        f"{successful + failed}"
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"Total chunks: {total_chunks}"
    )

    print(
        f"Chunks directory: "
        f"{CHUNKS_DIR}"
    )

    print("=" * 80)

    if embed:
        sync_embeddings()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Tecsack production document ingestion"
        )
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Print parsed elements and generated chunks"
        ),
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Process all registered documents "
            "regardless of change status"
        ),
    )

    parser.add_argument(
        "--no-embed",
        action="store_true",
        help=(
            "Only parse and chunk; do not sync "
            "embeddings into ChromaDB"
        ),
    )

    args = parser.parse_args()

    if args.rebuild:

        run_full_rebuild(
            debug=args.debug,
            embed=not args.no_embed,
        )

    else:

        run_ingestion(
            debug=args.debug,
            embed=not args.no_embed,
        )
