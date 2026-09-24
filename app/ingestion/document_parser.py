from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import re

import pymupdf
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document


# ---------------------------------------------------------------------
# Tesseract location
# ---------------------------------------------------------------------
#
# Use TESSERACT_CMD if set; otherwise fall back to the default
# Windows install path when tesseract is not on PATH.

def _configure_tesseract() -> None:
    import os
    import shutil

    configured = os.getenv("TESSERACT_CMD")

    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
        return

    if shutil.which("tesseract"):
        return

    default_windows_path = Path(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if default_windows_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(
            default_windows_path
        )


_configure_tesseract()


@dataclass
class DocumentElement:
    """
    One logical element extracted from a document.

    Examples:
        heading
        paragraph
        list
        table
    """

    element_type: str
    text: str
    page: int | None = None
    heading_level: int | None = None
    heading_path: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving readable text.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Zero-width characters (common in Google Docs PDF exports)
    # break pattern matching such as numbered-heading detection.
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces around lines
    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def detect_markdown_heading(line: str) -> tuple[int, str] | None:
    """
    Detect Markdown headings.

    Example:
        # Introduction       -> level 1
        ## Architecture      -> level 2
        ### Database         -> level 3
    """

    match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)

    if not match:
        return None

    level = len(match.group(1))
    title = match.group(2).strip()

    return level, title


def is_list_line(line: str) -> bool:
    """
    Detect common bullet/numbered list formats.
    """

    return bool(
        re.match(
            r"^("
            r"[-*•]\s+"
            r"|\d+[.)]\s+"
            r"|[a-zA-Z][.)]\s+"
            r")",
            line,
        )
    )


# ---------------------------------------------------------------------
# Heading hierarchy
# ---------------------------------------------------------------------

class HeadingStack(list):
    """
    List of heading texts that also remembers each heading's level.

    .copy() still returns a plain list of heading texts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.levels: list[int] = []


def update_heading_path(
    heading_stack: list[str],
    heading_level: int,
    heading_text: str,
) -> list[str]:
    """
    Maintain the current heading hierarchy.

    Example:

        H1: Project
        H2: Architecture
        H3: Database

    becomes:

        ["Project", "Architecture", "Database"]

    With a HeadingStack, nesting is decided by heading level, so
    skipped levels do not shift the hierarchy:

        H1 ABOUT US, H3 6.1 Healthcare, H4 6.1.1 ..., H4 6.1.2 ...

    gives ["ABOUT US", "6.1 Healthcare", "6.1.2 ..."].
    """

    levels = getattr(heading_stack, "levels", None)

    if levels is None:
        # Plain list: position == level.
        heading_stack[:] = heading_stack[: heading_level - 1]
        heading_stack.append(heading_text)
        return heading_stack.copy()

    # Remove same-level and deeper headings.
    while levels and levels[-1] >= heading_level:
        levels.pop()
        heading_stack.pop()

    heading_stack.append(heading_text)
    levels.append(heading_level)

    return heading_stack.copy()


def is_repeated_page_title(
    heading_stack: list[str],
    heading_level: int,
    heading_text: str,
) -> bool:
    """
    True for a running page header such as "ABOUT US" that is
    repeated at the top of every page. Treating it as a new
    level-1 heading would wipe the section context at each
    page break.
    """

    levels = getattr(heading_stack, "levels", None)

    return bool(
        levels
        and heading_level == 1
        and levels[0] == 1
        and heading_stack[0].casefold() == heading_text.casefold()
    )


# ---------------------------------------------------------------------
# TXT parser
# ---------------------------------------------------------------------

def parse_txt(file_path: Path) -> list[DocumentElement]:
    """
    Parse plain text files.

    Blank-line separated sections are treated as paragraphs.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    text = clean_text(text)

    if not text:
        return []

    elements: list[DocumentElement] = []

    heading_stack: list[str] = HeadingStack()

    paragraphs = re.split(r"\n\s*\n", text)

    for paragraph in paragraphs:
        paragraph = clean_text(paragraph)

        if not paragraph:
            continue

        lines = paragraph.split("\n")

        # Detect markdown headings inside TXT as well.
        if len(lines) == 1:
            heading = detect_markdown_heading(lines[0])

            if heading:
                level, title = heading

                path = update_heading_path(
                    heading_stack,
                    level,
                    title,
                )

                elements.append(
                    DocumentElement(
                        element_type="heading",
                        text=title,
                        heading_level=level,
                        heading_path=path,
                    )
                )

                continue

        element_type = (
            "list"
            if any(is_list_line(line) for line in lines)
            else "paragraph"
        )

        elements.append(
            DocumentElement(
                element_type=element_type,
                text=paragraph,
                heading_path=heading_stack.copy(),
            )
        )

    return elements


# ---------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------

def parse_markdown(file_path: Path) -> list[DocumentElement]:
    """
    Parse Markdown while preserving heading hierarchy.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = text.splitlines()

    elements: list[DocumentElement] = []

    heading_stack: list[str] = HeadingStack()

    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_buffer:
            return

        paragraph = clean_text(
            "\n".join(paragraph_buffer)
        )

        paragraph_buffer.clear()

        if not paragraph:
            return

        element_type = (
            "list"
            if any(
                is_list_line(line)
                for line in paragraph.split("\n")
            )
            else "paragraph"
        )

        elements.append(
            DocumentElement(
                element_type=element_type,
                text=paragraph,
                heading_path=heading_stack.copy(),
            )
        )

    for line in lines:
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            continue

        heading = detect_markdown_heading(stripped)

        if heading:
            flush_paragraph()

            level, title = heading

            path = update_heading_path(
                heading_stack,
                level,
                title,
            )

            elements.append(
                DocumentElement(
                    element_type="heading",
                    text=title,
                    heading_level=level,
                    heading_path=path,
                )
            )

            continue

        # Preserve Markdown table rows together.
        if stripped.startswith("|"):
            flush_paragraph()

            table_lines = [stripped]

            elements.append(
                DocumentElement(
                    element_type="table",
                    text="\n".join(table_lines),
                    heading_path=heading_stack.copy(),
                )
            )

            continue

        paragraph_buffer.append(stripped)

    flush_paragraph()

    return elements


# ---------------------------------------------------------------------
# DOCX parser
# ---------------------------------------------------------------------

def get_docx_heading_level(paragraph) -> int | None:
    """
    Detect Word heading style.

    Heading 1 -> 1
    Heading 2 -> 2
    etc.
    """

    style_name = ""

    try:
        style_name = paragraph.style.name or ""
    except Exception:
        pass

    match = re.search(
        r"heading\s*(\d+)",
        style_name,
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return None


def parse_docx(file_path: Path) -> list[DocumentElement]:
    """
    Parse DOCX while preserving headings, paragraphs,
    lists and tables.
    """

    document = Document(file_path)

    elements: list[DocumentElement] = []

    heading_stack: list[str] = HeadingStack()

    # -------------------------------------------------------------
    # Paragraphs
    # -------------------------------------------------------------

    for paragraph in document.paragraphs:
        text = clean_text(paragraph.text)

        if not text:
            continue

        heading_level = get_docx_heading_level(paragraph)

        if heading_level is not None:
            path = update_heading_path(
                heading_stack,
                heading_level,
                text,
            )

            elements.append(
                DocumentElement(
                    element_type="heading",
                    text=text,
                    heading_level=heading_level,
                    heading_path=path,
                )
            )

            continue

        style_name = ""

        try:
            style_name = paragraph.style.name or ""
        except Exception:
            pass

        is_list = (
            "list" in style_name.lower()
            or is_list_line(text)
        )

        elements.append(
            DocumentElement(
                element_type="list" if is_list else "paragraph",
                text=text,
                heading_path=heading_stack.copy(),
            )
        )

    # -------------------------------------------------------------
    # Tables
    # -------------------------------------------------------------

    for table_index, table in enumerate(document.tables):

        rows: list[str] = []

        for row in table.rows:
            cells: list[str] = []

            for cell in row.cells:
                cell_text = clean_text(cell.text)

                cells.append(cell_text)

            rows.append(" | ".join(cells))

        table_text = "\n".join(rows).strip()

        if not table_text:
            continue

        elements.append(
            DocumentElement(
                element_type="table",
                text=table_text,
                heading_path=heading_stack.copy(),
                metadata={
                    "table_index": table_index,
                },
            )
        )

    return elements


# ---------------------------------------------------------------------
# PDF heading detection
# ---------------------------------------------------------------------

def _get_pdf_block_text(block: dict[str, Any]) -> str:
    """
    Extract text from a PyMuPDF text block.
    """

    lines = block.get("lines", [])

    output_lines: list[str] = []

    for line in lines:
        spans = line.get("spans", [])

        line_text = "".join(
            span.get("text", "")
            for span in spans
        )

        if line_text.strip():
            output_lines.append(line_text.strip())

    return clean_text("\n".join(output_lines))


def _get_pdf_block_font_size(block: dict[str, Any]) -> float:
    """
    Return the largest font size in a PDF block.
    """

    sizes: list[float] = []

    for line in block.get("lines", []):
        for span in line.get("spans", []):
            size = span.get("size")

            if isinstance(size, (int, float)):
                sizes.append(float(size))

    return max(sizes) if sizes else 0.0


def _is_pdf_bold(block: dict[str, Any]) -> bool:
    """
    Detect whether the majority of useful spans are bold.
    """

    total = 0
    bold = 0

    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text = span.get("text", "").strip()

            if not text:
                continue

            total += 1

            font_name = span.get("font", "").lower()

            if (
                "bold" in font_name
                or "black" in font_name
                or "demi" in font_name
            ):
                bold += 1

    if total == 0:
        return False

    return bold / total >= 0.5


def get_numbered_heading_level(text: str) -> int | None:
    """
    Derive heading level from section numbering.

        "4. Leadership Team"        -> 2
        "5.1 Software Engineers"    -> 3
        "6.1.1 Telemedicine & EHR"  -> 4

    Level 1 is left for the unnumbered document title
    (e.g. "ABOUT US"). Returns None for unnumbered headings.
    Components are limited to 1-2 digits so years such as
    "2025 Holidays" are not treated as section numbers.
    """

    first_line = text.split("\n", 1)[0].strip()

    match = re.match(
        r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+[A-Za-z]",
        first_line,
    )

    if not match:
        return None

    return len(match.group(1).split(".")) + 1


def parse_pdf(file_path: Path) -> list[DocumentElement]:
    """
    Parse PDF pages using multiple extraction strategies.

    Strategy:

        1. PyMuPDF text extraction
        2. pdfplumber table extraction
        3. Tesseract OCR fallback for scanned/image PDFs

    The parser returns the same DocumentElement structure
    regardless of which extraction method was used.
    """

    document = pymupdf.open(file_path)

    elements: list[DocumentElement] = []

    heading_stack: list[str] = HeadingStack()

    try:

        # Open once for table extraction.
        plumber_pdf = pdfplumber.open(file_path)

        try:

            for page_index in range(len(document)):

                page = document[page_index]

                page_number = page_index + 1

                page_elements: list[DocumentElement] = []

                # =================================================
                # 1. NORMAL PDF TEXT EXTRACTION
                # =================================================

                page_dict = page.get_text("dict")

                blocks = page_dict.get("blocks", [])

                # -------------------------------------------------
                # Determine common body font size.
                # -------------------------------------------------

                all_font_sizes: list[float] = []

                for block in blocks:

                    if block.get("type") != 0:
                        continue

                    for line in block.get("lines", []):

                        for span in line.get("spans", []):

                            size = span.get("size")

                            if isinstance(size, (int, float)):
                                all_font_sizes.append(float(size))

                body_font_size = 0.0

                if all_font_sizes:

                    sorted_sizes = sorted(all_font_sizes)

                    middle = len(sorted_sizes) // 2

                    if len(sorted_sizes) % 2 == 0:

                        body_font_size = (
                            sorted_sizes[middle - 1]
                            + sorted_sizes[middle]
                        ) / 2

                    else:

                        body_font_size = sorted_sizes[middle]

                # -------------------------------------------------
                # Extract normal text blocks.
                # -------------------------------------------------

                for block in blocks:

                    if block.get("type") != 0:
                        continue

                    text = _get_pdf_block_text(block)

                    if not text:
                        continue

                    font_size = _get_pdf_block_font_size(block)

                    is_bold = _is_pdf_bold(block)

                    line_count = len(
                        block.get("lines", [])
                    )

                    # ---------------------------------------------
                    # Heading detection
                    # ---------------------------------------------

                    is_heading = False

                    if body_font_size > 0 and font_size > 0:

                        larger_than_body = (
                            font_size >= body_font_size * 1.20
                        )

                        short_text = len(text) <= 180

                        if (
                            larger_than_body
                            and short_text
                            and line_count <= 3
                        ):
                            is_heading = True

                        elif (
                            is_bold
                            and short_text
                            and line_count <= 2
                            and font_size >= body_font_size * 1.05
                        ):
                            is_heading = True

                    if is_heading:

                        ratio = (
                            font_size / body_font_size
                            if body_font_size
                            else 1
                        )

                        numbered_level = get_numbered_heading_level(
                            text
                        )

                        # Section numbers are more reliable than
                        # font size ("5." must not nest under "4.").
                        if numbered_level is not None:
                            level = numbered_level
                        elif ratio >= 1.7:
                            level = 1
                        elif ratio >= 1.4:
                            level = 2
                        else:
                            level = 3

                        if is_repeated_page_title(
                            heading_stack,
                            level,
                            text,
                        ):
                            continue

                        path = update_heading_path(
                            heading_stack,
                            level,
                            text,
                        )

                        page_elements.append(
                            DocumentElement(
                                element_type="heading",
                                text=text,
                                page=page_number,
                                heading_level=level,
                                heading_path=path,
                                metadata={
                                    "extraction_method": "pymupdf",
                                    "font_size": round(
                                        font_size,
                                        2,
                                    ),
                                    "bold": is_bold,
                                },
                            )
                        )

                    else:

                        element_type = (
                            "list"
                            if any(
                                is_list_line(line)
                                for line in text.split("\n")
                            )
                            else "paragraph"
                        )

                        page_elements.append(
                            DocumentElement(
                                element_type=element_type,
                                text=text,
                                page=page_number,
                                heading_path=heading_stack.copy(),
                                metadata={
                                    "extraction_method": "pymupdf",
                                },
                            )
                        )

                # =================================================
                # 2. NATIVE PDF TABLE EXTRACTION
                # =================================================

                plumber_page = plumber_pdf.pages[page_index]

                try:

                    tables = plumber_page.extract_tables()

                    for table_index, table in enumerate(tables):

                        if not table:
                            continue

                        rows: list[str] = []

                        for row in table:

                            if not row:
                                continue

                            cleaned_cells: list[str] = []

                            for cell in row:

                                if cell is None:
                                    cell_text = ""
                                else:
                                    cell_text = clean_text(
                                        str(cell)
                                    )

                                cleaned_cells.append(cell_text)

                            # Ignore completely empty rows.
                            if not any(cleaned_cells):
                                continue

                            rows.append(
                                " | ".join(cleaned_cells)
                            )

                        if not rows:
                            continue

                        table_text = "\n".join(rows).strip()

                        if not table_text:
                            continue

                        page_elements.append(
                            DocumentElement(
                                element_type="table",
                                text=table_text,
                                page=page_number,
                                heading_path=heading_stack.copy(),
                                metadata={
                                    "extraction_method": "pdfplumber",
                                    "table_index": table_index,
                                },
                            )
                        )

                except Exception as exc:

                    print(
                        f"WARNING: Table extraction failed "
                        f"on page {page_number}: {exc}"
                    )

                # =================================================
                # 3. OCR FALLBACK
                # =================================================
                #
                # If PyMuPDF and pdfplumber found nothing,
                # assume this is an image/scanned page.
                # =================================================

                if not page_elements:

                    print(
                        f"INFO: No text/table found on "
                        f"page {page_number}. "
                        f"Running OCR..."
                    )

                    pix = page.get_pixmap(
                        matrix=pymupdf.Matrix(3, 3),
                        alpha=False,
                    )

                    image = Image.frombytes(
                        "RGB",
                        [pix.width, pix.height],
                        pix.samples,
                    )

                    ocr_text = pytesseract.image_to_string(
                        image,
                        config="--psm 4",
                    )

                    ocr_text = clean_text(ocr_text)

                    if ocr_text:

                        page_elements.append(
                            DocumentElement(
                                element_type="paragraph",
                                text=ocr_text,
                                page=page_number,
                                heading_path=heading_stack.copy(),
                                metadata={
                                    "extraction_method": "tesseract_ocr",
                                },
                            )
                        )

                # -------------------------------------------------
                # Add page elements to final result.
                # -------------------------------------------------

                elements.extend(page_elements)

        finally:

            plumber_pdf.close()

    finally:

        document.close()

    return elements


# ---------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------

def parse_document(
    file_path: str | Path,
) -> list[DocumentElement]:
    """
    Parse a supported document.

    Supported:
        PDF
        DOCX
        TXT
        MD
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document does not exist: {path}"
        )

    extension = path.suffix.lower()

    if extension == ".pdf":
        return parse_pdf(path)

    if extension == ".docx":
        return parse_docx(path)

    if extension == ".txt":
        return parse_txt(path)

    if extension == ".md":
        return parse_markdown(path)

    if extension == ".doc":
        raise ValueError(
            "Legacy .doc files are not directly supported. "
            "Convert them to .docx or PDF before ingestion."
        )

    raise ValueError(
        f"Unsupported document type: {extension}"
    )


# ---------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------

def print_elements(
    elements: list[DocumentElement],
) -> None:

    print()
    print("=" * 80)
    print("PARSED DOCUMENT")
    print("=" * 80)

    for index, element in enumerate(elements):

        print()
        print(f"[{index}]")
        print(f"Type: {element.element_type}")
        print(f"Page: {element.page}")
        print(f"Heading level: {element.heading_level}")
        print(f"Heading path: {element.heading_path}")
        print("Text:")
        print(element.text)

    print()
    print("=" * 80)