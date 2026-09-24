
from __future__ import annotations

from typing import Any


def _get_page_range(metadata: dict[str, Any]) -> str:
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")

    if page_start is None:
        return "Unknown"

    if page_end is None or page_end == page_start:
        return str(page_start)

    return f"{page_start}-{page_end}"


def build_context(
    results: list[dict[str, Any]],
) -> str:
    """
    Convert retrieval results into structured context
    for the LLM.

    The retrieval result's metadata is preserved so the
    generated answer can refer to the original document.
    """

    if not results:
        return "No relevant documents were found."

    context_parts: list[str] = []

    for index, result in enumerate(results, start=1):

        metadata = result.get("metadata", {})

        file_name = metadata.get(
            "file_name",
            "Unknown file",
        )

        section = metadata.get(
            "heading_path",
            "Unknown section",
        )

        pages = _get_page_range(metadata)

        text = result.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        source = f"""
[Source {index}]
File: {file_name}
Page: {pages}
Section: {section}

{text}
""".strip()

        context_parts.append(source)

    if not context_parts:
        return "No relevant document content was found."

    return "\n\n".join(context_parts)

