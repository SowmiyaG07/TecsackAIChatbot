from __future__ import annotations

import json
import re
from typing import Any

from dotenv import load_dotenv
from ollama import Client


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

import os


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.1:latest",
)

MAX_EXPANDED_QUERIES = int(
    os.getenv("MAX_EXPANDED_QUERIES", "5")
)


# ---------------------------------------------------------
# Client
# ---------------------------------------------------------

ollama_client = Client(
    host=OLLAMA_BASE_URL
)


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

QUERY_EXPANSION_PROMPT = """
You are a search query expansion component for a company knowledge base.

Your task is to generate alternative search queries that help retrieve
documents relevant to the user's original question.

Rules:

1. Preserve the original meaning.
2. Do not answer the question.
3. Do not invent facts, names, companies, products, or topics.
4. Generate short, search-friendly queries.
5. Keep specific entities from the question (project names, product
   names, people, years), but do NOT add the company name
   "Tecsack" / "Tecsack Software". Every document belongs to this
   company, so the company name does not help find the right one.
6. Use different but semantically related wording, including the
   words the documents themselves are likely to use.
   ONLY IF the question asks about people, staff, team members or
   leadership, include job-title words such as CEO, CTO, founder,
   team lead or developers. Otherwise never add job titles.
7. Prefer queries that could match headings, sections, and document content.
8. Return ONLY valid JSON.
9. The JSON must contain a key named "queries".
10. "queries" must be a JSON array of strings.
11. Generate at most 5 queries.

User question:
{question}

Query intent:
{intent}

Return:
{{
    "queries": [
        "query 1",
        "query 2",
        "query 3"
    ]
}}
"""


# ---------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------

def extract_json_object(
    text: str,
) -> dict[str, Any] | None:
    """
    Extract the first JSON object from an LLM response.
    """

    text = text.strip()

    # Direct JSON
    try:
        value = json.loads(text)

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        pass

    # JSON inside markdown code block or surrounding text
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if not match:
        return None

    try:
        value = json.loads(
            match.group(0)
        )

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        return None

    return None


# ---------------------------------------------------------
# Query cleanup
# ---------------------------------------------------------

def clean_queries(
    queries: list[Any],
    original_question: str,
) -> list[str]:
    """
    Clean, validate, and deduplicate expanded queries.
    """

    cleaned: list[str] = []

    original_normalized = (
        original_question
        .strip()
        .casefold()
    )

    for query in queries:

        if not isinstance(query, str):
            continue

        query = query.strip()

        if not query:
            continue

        # Remove surrounding quotes if the model added them.
        query = query.strip("\"'")

        if not query:
            continue

        normalized = query.casefold()

        # Avoid duplicates.
        if normalized in {
            item.casefold()
            for item in cleaned
        }:
            continue

        cleaned.append(query)

    # Always keep the original question.
    if original_question.strip():
        if original_normalized not in {
            item.casefold()
            for item in cleaned
        }:
            cleaned.insert(
                0,
                original_question.strip(),
            )

    return cleaned[
        :MAX_EXPANDED_QUERIES
    ]


# ---------------------------------------------------------
# Query expansion
# ---------------------------------------------------------

def expand_query(
    question: str,
    intent: str = "general",
) -> list[str]:
    """
    Generate multiple search queries from the original question.

    The original question is always included.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    prompt = QUERY_EXPANSION_PROMPT.format(
        question=question,
        intent=intent,
    )

    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0,
        },
    )

    content = response["message"]["content"]

    parsed = extract_json_object(
        content
    )

    if not parsed:
        print(
            "Warning: Query expansion returned "
            "invalid JSON. Using original question."
        )

        return [question]

    queries = parsed.get(
        "queries",
        [],
    )

    if not isinstance(queries, list):
        return [question]

    expanded_queries = clean_queries(
        queries=queries,
        original_question=question,
    )

    if not expanded_queries:
        return [question]

    return expanded_queries


# ---------------------------------------------------------
# Debug output
# ---------------------------------------------------------

def print_expanded_queries(
    question: str,
    queries: list[str],
) -> None:

    print()
    print("=" * 80)
    print("QUERY EXPANSION")
    print("=" * 80)

    print(
        f"Original question: {question}"
    )

    print(
        f"Expanded queries : {len(queries)}"
    )

    print()

    for index, query in enumerate(
        queries,
        start=1,
    ):
        print(
            f"{index}. {query}"
        )