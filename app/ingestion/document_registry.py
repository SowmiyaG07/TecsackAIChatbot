
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

# Project root:
# TecsackAgent/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Source document directory
DATA_DIR = PROJECT_ROOT / "data"

# Registry location
STORAGE_DIR = PROJECT_ROOT / "storage"
REGISTRY_FILE = STORAGE_DIR / "document_registry.json"

# File types that the ingestion pipeline supports.
# Add more extensions when you add new parsers.
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".md",
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def utc_now() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate SHA-256 hash of a file.

    The file is read in chunks so large documents do not need
    to be loaded completely into memory.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def normalize_name(value: str) -> str:
    """
    Normalize a folder/file name so it can safely be used
    as part of a document ID.
    """

    value = value.strip().lower()

    cleaned = []

    for char in value:
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")

    normalized = "".join(cleaned)

    # Avoid repeated underscores
    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


def generate_document_id(relative_path: Path) -> str:
    """
    Generate a stable document ID based on the relative path.

    Example:

        Project/giftcard/api.pdf

    becomes:

        doc_project_giftcard_api
    """

    path_without_extension = relative_path.with_suffix("")

    parts = [
        normalize_name(part)
        for part in path_without_extension.parts
    ]

    return "doc_" + "_".join(parts)


# ============================================================
# CATEGORY / PROJECT EXTRACTION
# ============================================================

def extract_location_metadata(relative_path: Path) -> dict[str, str | None]:
    """
    Convert folder hierarchy into metadata.

    Examples:

        company/about.pdf

        {
            "top_level": "company",
            "project": None
        }

        HR/leave_policy.pdf

        {
            "top_level": "hr",
            "project": None
        }

        Project/giftcard/api.pdf

        {
            "top_level": "project",
            "project": "giftcard"
        }
    """

    parts = list(relative_path.parts)

    if not parts:
        return {
            "top_level": None,
            "project": None,
        }

    top_level = parts[0].lower()

    project = None

    if top_level == "project" and len(parts) >= 3:
        project = parts[1]

    return {
        "top_level": top_level,
        "project": project,
    }


# ============================================================
# DOCUMENT DISCOVERY
# ============================================================

def discover_documents(data_dir: Path) -> list[Path]:
    """
    Recursively discover supported documents.

    Example:

        data/
            company/
                about.pdf
            HR/
                leave.pdf
            Project/
                giftcard/
                    api.pdf

    All supported files are returned.
    """

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}"
        )

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"Data path is not a directory: {data_dir}"
        )

    documents: list[Path] = []

    for path in data_dir.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        documents.append(path)

    # Sort so processing order is deterministic.
    documents.sort(
        key=lambda path: str(path).lower()
    )

    return documents


# ============================================================
# DOCUMENT METADATA
# ============================================================

def build_document_record(
    file_path: Path,
    data_dir: Path,
) -> dict[str, Any]:
    """
    Build metadata for one document.
    """

    relative_path = file_path.relative_to(data_dir)

    location_metadata = extract_location_metadata(
        relative_path
    )

    file_hash = calculate_sha256(file_path)

    stat = file_path.stat()

    document_id = generate_document_id(
        relative_path
    )

    record: dict[str, Any] = {
        "document_id": document_id,

        "file_name": file_path.name,

        "relative_path": relative_path.as_posix(),

        "file_extension": file_path.suffix.lower(),

        "file_size_bytes": stat.st_size,

        "file_hash": file_hash,

        "top_level": location_metadata["top_level"],

        "project": location_metadata["project"],

        "version": 1,

        "status": "active",

        "first_seen_at": utc_now(),

        "last_seen_at": utc_now(),

        "last_processed_at": None,
    }

    return record


# ============================================================
# REGISTRY STORAGE
# ============================================================

def load_registry() -> dict[str, Any]:
    """
    Load the document registry.

    If the registry does not exist yet, return an empty registry.
    """

    if not REGISTRY_FILE.exists():
        return {
            "documents": {}
        }

    try:
        with REGISTRY_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Registry file contains invalid JSON: "
            f"{REGISTRY_FILE}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            "Document registry must contain a JSON object."
        )

    data.setdefault("documents", {})

    return data


def save_registry(
    registry: dict[str, Any],
) -> None:
    """
    Save registry atomically.

    We first write to a temporary file and then replace the
    original registry.
    """

    STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = REGISTRY_FILE.with_suffix(
        ".tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            registry,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_file.replace(REGISTRY_FILE)


# ============================================================
# CHANGE DETECTION
# ============================================================

def detect_document_changes(
    discovered_documents: list[Path],
    registry: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Compare documents on disk against the existing registry.

    Returns:

        {
            "new": [...],
            "changed": [...],
            "unchanged": [...],
            "deleted": [...]
        }
    """

    registry_documents: dict[str, Any] = registry.get(
        "documents",
        {},
    )

    new_documents: list[dict[str, Any]] = []

    changed_documents: list[dict[str, Any]] = []

    unchanged_documents: list[dict[str, Any]] = []

    discovered_ids: set[str] = set()

    for file_path in discovered_documents:

        relative_path = file_path.relative_to(
            DATA_DIR
        )

        document_id = generate_document_id(
            relative_path
        )

        discovered_ids.add(document_id)

        current_hash = calculate_sha256(
            file_path
        )

        existing = registry_documents.get(
            document_id
        )

        # ----------------------------------------------------
        # NEW DOCUMENT
        # ----------------------------------------------------

        if existing is None:

            record = build_document_record(
                file_path,
                DATA_DIR,
            )

            record["status"] = "new"

            new_documents.append(record)

            continue

        # ----------------------------------------------------
        # CHANGED DOCUMENT
        # ----------------------------------------------------

        if existing.get("file_hash") != current_hash:

            record = build_document_record(
                file_path,
                DATA_DIR,
            )

            previous_version = existing.get(
                "version",
                1,
            )

            record["version"] = (
                previous_version + 1
            )

            record["status"] = "changed"

            record["first_seen_at"] = existing.get(
                "first_seen_at",
                utc_now(),
            )

            changed_documents.append(record)

            continue

        # ----------------------------------------------------
        # UNCHANGED DOCUMENT
        # ----------------------------------------------------

        existing["last_seen_at"] = utc_now()

        existing["status"] = "unchanged"

        unchanged_documents.append(existing)

    # --------------------------------------------------------
    # DELETED DOCUMENTS
    # --------------------------------------------------------

    deleted_documents: list[dict[str, Any]] = []

    for document_id, existing in registry_documents.items():

        if document_id not in discovered_ids:

            deleted_record = dict(existing)

            deleted_record["status"] = "deleted"

            deleted_record["last_seen_at"] = utc_now()

            deleted_documents.append(
                deleted_record
            )

    return {
        "new": new_documents,
        "changed": changed_documents,
        "unchanged": unchanged_documents,
        "deleted": deleted_documents,
    }


# ============================================================
# REGISTRY UPDATE
# ============================================================

def update_registry(
    changes: dict[str, list[dict[str, Any]]],
    registry: dict[str, Any],
) -> None:
    """
    Update registry using detected document changes.
    """

    documents = registry.setdefault(
        "documents",
        {},
    )

    # --------------------------------------------------------
    # NEW
    # --------------------------------------------------------

    for record in changes["new"]:

        record["status"] = "active"

        documents[
            record["document_id"]
        ] = record

    # --------------------------------------------------------
    # CHANGED
    # --------------------------------------------------------

    for record in changes["changed"]:

        record["status"] = "active"

        documents[
            record["document_id"]
        ] = record

    # --------------------------------------------------------
    # UNCHANGED
    # --------------------------------------------------------

    for record in changes["unchanged"]:

        record["status"] = "active"

        documents[
            record["document_id"]
        ] = record

    # --------------------------------------------------------
    # DELETED
    # --------------------------------------------------------

    for record in changes["deleted"]:

        document_id = record[
            "document_id"
        ]

        if document_id in documents:

            documents[
                document_id
            ]["status"] = "deleted"

            documents[
                document_id
            ]["last_seen_at"] = utc_now()


# ============================================================
# INGESTION SCAN
# ============================================================

def scan_documents() -> dict[str, list[dict[str, Any]]]:
    """
    Main Step 1 operation.

    1. Discover documents.
    2. Load existing registry.
    3. Detect changes.
    4. Update registry.
    5. Save registry.
    """

    logger.info(
        "Starting document discovery..."
    )

    logger.info(
        "Data directory: %s",
        DATA_DIR,
    )

    documents = discover_documents(
        DATA_DIR
    )

    logger.info(
        "Discovered %d supported documents.",
        len(documents),
    )

    registry = load_registry()

    changes = detect_document_changes(
        documents,
        registry,
    )

    update_registry(
        changes,
        registry,
    )

    save_registry(
        registry
    )

    return changes


# ============================================================
# REPORT
# ============================================================

def print_report(
    changes: dict[str, list[dict[str, Any]]],
) -> None:
    """
    Print a simple ingestion report.
    """

    print()
    print("=" * 70)
    print("DOCUMENT INGESTION REPORT")
    print("=" * 70)

    print(
        f"NEW       : {len(changes['new'])}"
    )

    print(
        f"CHANGED   : {len(changes['changed'])}"
    )

    print(
        f"UNCHANGED : {len(changes['unchanged'])}"
    )

    print(
        f"DELETED   : {len(changes['deleted'])}"
    )

    print("=" * 70)

    if changes["new"]:

        print()
        print("NEW DOCUMENTS")
        print("-" * 70)

        for record in changes["new"]:

            print(
                f"{record['document_id']} | "
                f"{record['relative_path']}"
            )

    if changes["changed"]:

        print()
        print("CHANGED DOCUMENTS")
        print("-" * 70)

        for record in changes["changed"]:

            print(
                f"{record['document_id']} | "
                f"version={record['version']} | "
                f"{record['relative_path']}"
            )

    if changes["deleted"]:

        print()
        print("DELETED DOCUMENTS")
        print("-" * 70)

        for record in changes["deleted"]:

            print(
                f"{record['document_id']} | "
                f"{record['relative_path']}"
            )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        changes = scan_documents()

        print_report(changes)

        print(
            f"Registry saved to: {REGISTRY_FILE}"
        )

    except Exception as exc:

        logger.exception(
            "Document ingestion scan failed: %s",
            exc,
        )

        raise
