from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import re

from app.ingestion.document_parser import DocumentElement


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET_CHUNK_TOKENS = 500
MAX_CHUNK_TOKENS = 650
OVERLAP_TOKENS = 60


# ---------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------

try:
    import tiktoken

    _encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_encoding.encode(text))

except Exception:

    def count_tokens(text: str) -> int:
        """
        Approximate token count.

        Rough fallback:
            1 token ~= 4 characters
        """

        if not text:
            return 0

        return max(
            1,
            len(text) // 4,
        )


# ---------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------

@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str

    chunk_index: int

    token_count: int

    page_start: int | None
    page_end: int | None

    section: str | None
    heading_path: list[str]

    content_type: str

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------

def normalize_chunk_text(text: str) -> str:
    """
    Normalize extracted document text.

    Handles:
    - null characters
    - zero-width Unicode characters
    - repeated spaces
    - excessive blank lines
    """

    if not text:
        return ""

    # Remove characters commonly introduced by PDF extraction.
    text = text.replace("\x00", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\ufeff", "")

    # Normalize whitespace.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def is_standalone_noise(text: str) -> bool:
    """
    Return True for standalone text fragments that should not
    become their own chunks.
    """
    normalized = normalize_chunk_text(text).casefold()

    if not normalized:
        return True

    # Document/company branding that appears as a standalone element
    noise_values = {
        "tecsack software",
        "tecsack",
    }

    if normalized in noise_values:
        return True

    return False


def split_into_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter.

    This intentionally avoids heavyweight NLP dependencies.
    """

    text = normalize_chunk_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_long_text(
    text: str,
    max_tokens: int,
) -> list[str]:
    """
    Split a paragraph that is larger than MAX_CHUNK_TOKENS.

    First attempt:
        sentence boundaries

    If a single sentence is still too large:
        word boundaries
    """

    text = normalize_chunk_text(text)

    if count_tokens(text) <= max_tokens:
        return [text]

    sentences = split_into_sentences(text)

    if not sentences:
        return split_by_words(
            text,
            max_tokens,
        )

    parts: list[str] = []

    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:

        sentence_tokens = count_tokens(sentence)

        if sentence_tokens > max_tokens:

            if current:
                parts.append(
                    " ".join(current)
                )

                current = []
                current_tokens = 0

            parts.extend(
                split_by_words(
                    sentence,
                    max_tokens,
                )
            )

            continue

        if (
            current
            and current_tokens + sentence_tokens > max_tokens
        ):
            parts.append(
                " ".join(current)
            )

            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        parts.append(
            " ".join(current)
        )

    return parts


def split_by_words(
    text: str,
    max_tokens: int,
) -> list[str]:

    words = text.split()

    parts: list[str] = []

    current: list[str] = []
    current_tokens = 0

    for word in words:

        word_tokens = count_tokens(word + " ")

        if (
            current
            and current_tokens + word_tokens > max_tokens
        ):
            parts.append(
                " ".join(current)
            )

            current = []
            current_tokens = 0

        current.append(word)
        current_tokens += word_tokens

    if current:
        parts.append(
            " ".join(current)
        )

    return parts


# ---------------------------------------------------------------------
# Heading formatting
# ---------------------------------------------------------------------

def format_heading_path(
    heading_path: list[str] | None,
) -> str:

    if not heading_path:
        return ""

    return " > ".join(
        item.strip()
        for item in heading_path
        if item.strip()
    )


def normalize_heading_path(
    heading_path: list[str] | None,
) -> list[str]:
    """
    Normalize every heading in the heading path.
    """

    if not heading_path:
        return []

    return [
        normalize_chunk_text(item)
        for item in heading_path
        if normalize_chunk_text(item)
    ]

# ---------------------------------------------------------------------
# Element preparation
# ---------------------------------------------------------------------


@dataclass
class PreparedElement:
    text: str
    element_type: str

    page: int | None

    heading_path: list[str]

    metadata: dict[str, Any]


def prepare_elements(
    elements: list[DocumentElement],
) -> list[PreparedElement]:

    prepared: list[PreparedElement] = []

    for element in elements:

        text = normalize_chunk_text(element.text)

        if not text:
            continue

        # Skip standalone branding/noise
        if is_standalone_noise(text):
            continue

        # -------------------------------------------------------------
        # Heading
        # -------------------------------------------------------------

        if element.element_type == "heading":

            prepared.append(
                PreparedElement(
                    text=text,
                    element_type="heading",
                    page=element.page,
                    heading_path=normalize_heading_path(
                        element.heading_path
                    ) or [text],
                    metadata=(
                        element.metadata
                        or {}
                    ),
                )
            )

            continue

        # -------------------------------------------------------------
        # Long paragraph/list/table
        # -------------------------------------------------------------

        pieces = split_long_text(
            text,
            MAX_CHUNK_TOKENS,
        )

        for piece in pieces:

            prepared.append(
                PreparedElement(
                    text=piece,
                    element_type=element.element_type,
                    page=element.page,
                    heading_path=normalize_heading_path(
                        element.heading_path
                    ),
                    metadata=(
                        element.metadata
                        or {}
                    ),
                )
            )

    return prepared

# ---------------------------------------------------------------------
# Chunk text construction
# ---------------------------------------------------------------------


def build_content_text(
    element: PreparedElement,
) -> str:
    return element.text


# ---------------------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------------------

def create_chunks(
    document_id: str,
    elements: list[DocumentElement],
    base_metadata: dict[str, Any] | None = None,
) -> list[DocumentChunk]:

    base_metadata = base_metadata or {}

    prepared = prepare_elements(elements)

    if not prepared:
        return []

    chunks: list[DocumentChunk] = []

    current_texts: list[str] = []
    current_tokens = 0

    current_pages: list[int] = []
    current_heading_path: list[str] = []

    current_content_types: list[str] = []

    def flush_chunk() -> None:

        nonlocal current_texts
        nonlocal current_tokens
        nonlocal current_pages
        nonlocal current_heading_path
        nonlocal current_content_types

        if not current_texts:
            return

        content_text = normalize_chunk_text(
            "\n\n".join(current_texts)
        )

        heading = format_heading_path(
            current_heading_path
        )

        if heading:
            text = normalize_chunk_text(
                f"Section: {heading}\n\n{content_text}"
            )
        else:
            text = content_text

        if not text:
            return

        chunk_index = len(chunks)

        page_start = (
            min(current_pages)
            if current_pages
            else None
        )

        page_end = (
            max(current_pages)
            if current_pages
            else None
        )

        content_type = determine_content_type(
            current_content_types
        )

        section = (
            current_heading_path[-1]
            if current_heading_path
            else None
        )

        chunk_id = (
            f"{document_id}_chunk_{chunk_index:05d}"
        )

        metadata = {
            **base_metadata,

            "chunk_index": chunk_index,

            "page_start": page_start,
            "page_end": page_end,

            "section": section,

            "heading_path": (
                current_heading_path.copy()
            ),

            "content_type": content_type,
        }

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=text,
                chunk_index=chunk_index,
                token_count=count_tokens(text),
                page_start=page_start,
                page_end=page_end,
                section=section,
                heading_path=(
                    current_heading_path.copy()
                ),
                content_type=content_type,
                metadata=metadata,
            )
        )

        current_texts = []
        current_tokens = 0
        current_pages = []
        current_heading_path = []
        current_content_types = []

    for element in prepared:

        # Heading elements are handled as section context below.
        if element.element_type == "heading":
            element_text = ""
            element_tokens = 0
        else:
            element_text = build_content_text(element)

        element_tokens = count_tokens(
            element_text
        )

        # -------------------------------------------------------------
        # Heading handling
        # -------------------------------------------------------------

        if element.element_type == "heading":

            # A heading is section context, not standalone content.
            #
            # If we already have meaningful content, finish the
            # previous section before starting the new section.
            if current_texts:
                flush_chunk()

        # Remember the heading for the next meaningful element.
            current_heading_path = (
                element.heading_path.copy()
            )

    # Do NOT add the heading itself to current_texts.
    #
    # The heading will be included as context when the
    # following meaningful content is processed.

            continue

        # -------------------------------------------------------------
        # Tables should remain intact whenever possible.
        # -------------------------------------------------------------

        if element.element_type == "table":

            if (
                current_texts
                and current_tokens + element_tokens
                > TARGET_CHUNK_TOKENS
            ):
                flush_chunk()

            current_texts.append(
                element_text
            )

            current_tokens += element_tokens

            if element.page is not None:
                current_pages.append(
                    element.page
                )

            current_content_types.append(
                "table"
            )

            continue

        # -------------------------------------------------------------
        # Normal paragraphs/lists
        # -------------------------------------------------------------

        if (
            current_texts
            and current_tokens + element_tokens
            > TARGET_CHUNK_TOKENS
        ):
            previous_text = (
                "\n\n".join(current_texts)
            )

            flush_chunk()

            # Add overlap from previous chunk.
            overlap = get_overlap_text(
                previous_text,
                OVERLAP_TOKENS,
            )

            if overlap:
                current_texts.append(
                    overlap
                )

                current_tokens = count_tokens(
                    overlap
                )

                # Continue the same section.
                current_heading_path = (
                    element.heading_path.copy()
                )

        current_texts.append(
            element_text
        )

        current_tokens += element_tokens

        if element.page is not None:
            current_pages.append(
                element.page
            )

        current_content_types.append(
            element.element_type
        )

    flush_chunk()

    return chunks


# ---------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------

def get_overlap_text(
    text: str,
    overlap_tokens: int,
) -> str:

    if not text:
        return ""

    words = text.split()

    if not words:
        return ""

    # Approximate number of words required.
    number_of_words = max(
        1,
        int(overlap_tokens / 1.3),
    )

    overlap_words = words[-number_of_words:]

    return " ".join(overlap_words)


def determine_content_type(
    content_types: list[str],
) -> str:

    if not content_types:
        return "text"

    if "table" in content_types:
        return "table"

    if "list" in content_types:
        return "list"

    return "text"


# ---------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------

def print_chunks(
    chunks: list[DocumentChunk],
) -> None:

    print()
    print("=" * 80)
    print(f"GENERATED CHUNKS: {len(chunks)}")
    print("=" * 80)

    for chunk in chunks:

        print()
        print("-" * 80)

        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Tokens: {chunk.token_count}")
        print(
            f"Pages: "
            f"{chunk.page_start} - {chunk.page_end}"
        )
        print(f"Section: {chunk.section}")
        print(
            f"Heading path: "
            f"{chunk.heading_path}"
        )
        print(
            f"Content type: "
            f"{chunk.content_type}"
        )

        print()
        print(chunk.text)

    print()
    print("=" * 80)
