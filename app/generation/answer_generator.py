
from __future__ import annotations

import os

from dotenv import load_dotenv
from ollama import Client

from app.observability import observed_chat


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.1:latest",
)

OLLAMA_TEMPERATURE = float(
    os.getenv(
        "OLLAMA_TEMPERATURE",
        "0",
    )
)


# ---------------------------------------------------------
# Ollama client
# ---------------------------------------------------------

ollama_client = Client(
    host=OLLAMA_BASE_URL
)


# ---------------------------------------------------------
# System prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are a company information assistant.

Your job is to answer the user's question using ONLY
the document context provided to you.

Rules:

1. Use only information present in the supplied context.

2. Do not use your own general knowledge.

3. Do not invent names, dates, roles, projects, policies,
   or other information.

4. If the answer cannot be found in the supplied context,
   say exactly:

   "I couldn't find that information in the available
   company documents."

5. Prefer the source that most directly answers the question.

6. If multiple sources contain relevant information,
   combine them only when they are consistent.

7. Keep the answer concise and easy to understand.

8. Include the relevant source file and page when possible.

9. Do not mention embeddings, vector search, BM25, RRF,
   retrieval, or internal implementation details.

10. Do not answer questions unrelated to the supplied
    company documents.
"""


# ---------------------------------------------------------
# Answer generation
# ---------------------------------------------------------

def format_history(
    history: list[dict[str, str]] | None,
) -> str:
    """
    Render recent conversation messages as a prompt section.

    Returns "" when there is no history, so the prompt is exactly
    the same as before conversation memory existed.
    """

    if not history:
        return ""

    lines = [
        f"{'User' if message['role'] == 'user' else 'Assistant'}: "
        f"{message['content'].strip()}"
        for message in history
    ]

    return (
        "RECENT CONVERSATION\n"
        "===================\n"
        "\n"
        "Use this only to understand what the current question refers\n"
        "to (for example \"he\", \"that project\", \"what about the CTO\").\n"
        "It is NOT a source of facts: take every fact from the DOCUMENT\n"
        "CONTEXT below. If the conversation and the documents disagree,\n"
        "the documents are correct.\n"
        "\n"
        + "\n\n".join(lines)
        + "\n\n\n"
    )


def generate_answer(
    question: str,
    context: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    Generate a grounded answer using the retrieved context.

    `history` is recent conversation (oldest first, Ollama chat
    format) from Redis. It only helps resolve references in the
    question; facts must come from `context`.
    """

    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if not context.strip():
        return (
            "I couldn't find that information in the "
            "available company documents."
        )

    user_prompt = f"""
{format_history(history)}DOCUMENT CONTEXT
================

{context}


USER QUESTION
=============

{question}


INSTRUCTIONS
============

Answer the question using only the document context above.

If the answer is present:
- Give a concise answer.
- Use the most relevant source.
- Mention the source file and page when appropriate.

If the answer is not present:
- Say that you couldn't find the information
  in the available company documents.

ANSWER:
"""

    response = observed_chat(
        ollama_client,
        "answer_generation",
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": OLLAMA_TEMPERATURE,
        },
    )

    answer = response["message"]["content"].strip()

    if not answer:
        return (
            "I couldn't generate an answer from the "
            "available company documents."
        )

    return answer

