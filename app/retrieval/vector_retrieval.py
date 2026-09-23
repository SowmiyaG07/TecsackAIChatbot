from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from ollama import Client


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

TOP_K = int(
    os.getenv("TOP_K", "5")
)

SIMILARITY_THRESHOLD = float(
    os.getenv("SIMILARITY_THRESHOLD", "0.2")
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

collection = chroma_client.get_collection(
    name=CHROMA_COLLECTION
)


# ---------------------------------------------------------
# Query intent
# ---------------------------------------------------------

def classify_query_intent(
    question: str,
) -> str:
    """
    Classify the question into a broad retrieval category.

    Returns:
        company
        hr
        project
        general
    """

    question_lower = question.lower().strip()

    # -----------------------------------------------------
    # HR-related questions
    # -----------------------------------------------------

    hr_keywords = [
        "hr",
        "human resource",
        "leave",
        "holiday",
        "salary",
        "payslip",
        "pay slip",
        "employee",
        "employment",
        "attendance",
        "policy",
        "policies",
        "benefit",
        "benefits",
        "payroll",
        "resolution",
    ]

    if any(
        keyword in question_lower
        for keyword in hr_keywords
    ):
        return "hr"

    # -----------------------------------------------------
    # Project-related questions
    # -----------------------------------------------------

    project_keywords = [
        "project",
        "projects",
        "peopleapps",
        "school feeding",
        "virtualterminal",
        "virtual terminal",
        "healthcare",
        "inspection",
        "iot",
        "giftcard",
        "gift card",
        "mobile platform",
        "enterprise",
    ]

    if any(
        keyword in question_lower
        for keyword in project_keywords
    ):
        return "project"

    # -----------------------------------------------------
    # Company-level questions
    # -----------------------------------------------------

    company_keywords = [
        "about tecsack",
        "about tecsack software",
        "what does tecsack do",
        "what does tecsack software do",
        "tecsack company",
        "tecsack software company",
        "company overview",
        "company profile",
        "company mission",
        "company vision",
        "services",
        "service",
        "who is tecsack",
        "tell me about tecsack",
        "explain about tecsack",
        "ceo",
        "founder",
        "leadership",
    ]

    if any(
        keyword in question_lower
        for keyword in company_keywords
    ):
        return "company"

    return "general"


# ---------------------------------------------------------
# Metadata filter
# ---------------------------------------------------------

def get_metadata_filter(
    intent: str,
) -> dict[str, Any] | None:
    """
    Convert query intent into a Chroma metadata filter.
    """

    if intent == "company":
        return {
            "top_level": "company"
        }

    if intent == "hr":
        return {
            "top_level": "hr"
        }

    if intent == "project":
        return {
            "top_level": "project"
        }

    return None


# ---------------------------------------------------------
# Query embedding
# ---------------------------------------------------------

def generate_query_embedding(
    question: str,
) -> list[float]:
    """
    Generate a 768-dimensional embedding
    for the user question.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    response = ollama_client.embed(
        model=EMBEDDING_MODEL,
        input=question,
    )

    embeddings = response["embeddings"]

    if not embeddings:
        raise RuntimeError(
            "No embedding returned for the question."
        )

    embedding = embeddings[0]

    print(
        f"Query embedding dimension: {len(embedding)}"
    )

    return embedding


# ---------------------------------------------------------
# Vector search
# ---------------------------------------------------------

def search_vectors(
    question: str,
    top_k: int = TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Search ChromaDB using vector similarity
    with query-aware metadata filtering.
    """

    query_embedding = generate_query_embedding(
        question
    )

    # -----------------------------------------------------
    # Determine query intent
    # -----------------------------------------------------

    intent = classify_query_intent(question)

    metadata_filter = get_metadata_filter(intent)

    print(
        f"Query intent      : {intent}"
    )

    print(
        f"Metadata filter   : {metadata_filter}"
    )

    # -----------------------------------------------------
    # Chroma search
    # -----------------------------------------------------

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if metadata_filter is not None:
        query_kwargs["where"] = metadata_filter

    results = collection.query(
        **query_kwargs
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    ids = results.get(
        "ids",
        [[]],
    )[0]

    retrieved: list[dict[str, Any]] = []

    for index, document in enumerate(documents):

        distance = distances[index]

        # Chroma uses cosine distance.
        #
        # distance = 0 → very similar
        # distance = 1 → less similar
        #
        # Convert to similarity.

        similarity = 1.0 - distance

        if similarity < similarity_threshold:
            continue

        retrieved.append(
            {
                "rank": len(retrieved) + 1,
                "id": ids[index],
                "text": document,
                "similarity": similarity,
                "distance": distance,
                "metadata": metadatas[index],
                "query_intent": intent,
            }
        )

    return retrieved


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

def print_results(
    question: str,
    results: list[dict[str, Any]],
) -> None:

    print()
    print("=" * 80)
    print("VECTOR SEARCH")
    print("=" * 80)

    print(
        f"Question: {question}"
    )

    print(
        f"Results : {len(results)}"
    )

    print()

    if not results:
        print(
            "No relevant chunks found."
        )
        return

    for result in results:

        print(
            "-" * 80
        )

        print(
            f"Rank       : "
            f"{result['rank']}"
        )

        print(
            f"Similarity : "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Distance   : "
            f"{result['distance']:.4f}"
        )

        print(
            f"Chunk ID   : "
            f"{result['id']}"
        )

        print(
            f"Intent     : "
            f"{result.get('query_intent', '')}"
        )

        metadata = result["metadata"]

        print(
            f"Document   : "
            f"{metadata.get('file_name', '')}"
        )

        print(
            f"Top Level  : "
            f"{metadata.get('top_level', '')}"
        )

        print(
            f"Project    : "
            f"{metadata.get('project', '')}"
        )

        print(
            f"Section    : "
            f"{metadata.get('section', '')}"
        )

        print(
            f"Page       : "
            f"{metadata.get('page', '')}"
        )

        print()

        print("Text:")

        print(
            result["text"]
        )


# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------

def main() -> None:

    print("=" * 80)
    print("VECTOR RETRIEVAL TEST")
    print("=" * 80)

    print(
        f"Chroma collection : "
        f"{CHROMA_COLLECTION}"
    )

    print(
        f"Stored vectors    : "
        f"{collection.count()}"
    )

    print(
        f"Embedding model   : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Top-K             : "
        f"{TOP_K}"
    )

    print(
        f"Threshold         : "
        f"{SIMILARITY_THRESHOLD}"
    )

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        print(
            "Question cannot be empty."
        )
        return

    results = search_vectors(
        question=question,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    print_results(
        question=question,
        results=results,
    )


if __name__ == "__main__":
    main()