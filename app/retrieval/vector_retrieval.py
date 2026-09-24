from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from app.retrieval.query_intent import (
    classify_query_intent,
    get_metadata_filter,
)
import chromadb
from dotenv import load_dotenv
from ollama import Client
from app.retrieval.query_expansion import (
    expand_query,
    print_expanded_queries,
)


from app.retrieval.bm25_retrieval import (
    get_bm25_index,
    search_bm25_index,
)
from app.embeddings.embedding_service import (
    format_query_for_embedding,
)

from app.generation.context_builder import build_context
from app.generation.answer_generator import generate_answer
from app.observability import record_retrieval, timed
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

RRF_K = int(
    os.getenv("RRF_K", "60")
)

RETRIEVAL_TOP_K = int(
    os.getenv("RETRIEVAL_TOP_K", "10")
)

# Names that appear in almost every chunk. In a query they carry no
# signal and pull the embedding towards generic "Tecsack Software
# is ..." chunks, so they are removed before vector search.
# Longest first so "tecsack software" is removed as a whole.
CORPUS_ENTITY_TERMS = sorted(
    (
        term.strip().lower()
        for term in os.getenv(
            "CORPUS_ENTITY_TERMS",
            "tecsack software private limited,"
            "tecsack software,tecsack",
        ).split(",")
        if term.strip()
    ),
    key=len,
    reverse=True,
)

# Words that do not make a query meaningful on their own.
QUERY_FILLER_WORDS = {
    "a", "about", "all", "an", "and", "are", "can", "describe",
    "detail", "details", "do", "does", "explain", "for", "give",
    "how", "i", "in", "info", "information", "is", "list", "me",
    "of", "on", "please", "s", "show", "tell", "the", "to",
    "what", "which", "who", "whom", "whose", "you",
}
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
# Query embedding
# ---------------------------------------------------------

def strip_corpus_entities(
    query: str,
) -> str:
    """
    Remove corpus-wide entity names (e.g. "Tecsack") from a query.

        "who are all team members of tecsack"
            -> "who are all team members"

    The original query is returned when nothing meaningful would
    remain, e.g. "tell me about tecsack".
    """

    stripped = query

    for term in CORPUS_ENTITY_TERMS:
        stripped = re.sub(
            rf"\b{re.escape(term)}(?:'s|’s)?\b",
            " ",
            stripped,
            flags=re.IGNORECASE,
        )

    # Tidy leftovers such as "of ?" or double spaces.
    stripped = re.sub(r"\s+", " ", stripped).strip()
    stripped = re.sub(r"\b(of|about|at|in|for)\s*([?.!]?)$", r"\2", stripped)
    stripped = re.sub(r"\s+([?.!,])", r"\1", stripped)
    stripped = stripped.strip()

    content_words = [
        word
        for word in re.findall(r"\w+", stripped.lower())
        if word not in QUERY_FILLER_WORDS
    ]

    if not content_words:
        return query

    return stripped


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

    with timed("embedding"):
        response = ollama_client.embed(
            model=EMBEDDING_MODEL,
            input=format_query_for_embedding(
                strip_corpus_entities(question)
            ),
        )

    embeddings = response["embeddings"]

    if not embeddings:
        raise RuntimeError(
            "No embedding returned for the question."
        )

    return embeddings[0]


# ---------------------------------------------------------
# Vector search
# ---------------------------------------------------------


def search_vectors(
    
    question: str,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:

    # ---------------------------------------------------------
    # 1. Determine query intent
    # ---------------------------------------------------------
    with timed("query_intent"):
        intent, justification = classify_query_intent(question)

        metadata_filter = get_metadata_filter(intent)

    print(f"Query intent      : {intent}")
    print(f"Justification     : {justification}")
    print(f"Metadata filter   : {metadata_filter}")

    # ---------------------------------------------------------
    # 2. Expand query
    # ---------------------------------------------------------
    with timed("query_expansion", emit_event=True) as span:
        expanded_queries = expand_query(
            question,
            intent=intent,
        )

        span["expanded_queries"] = len(expanded_queries)

    print(expanded_queries)

    # ---------------------------------------------------------
    # 3. BM25 index (cached per filter)
    # ---------------------------------------------------------
    with timed("bm25_index"):
        bm25_index, bm25_documents = get_bm25_index(
            collection,
            metadata_filter=metadata_filter,
        )

    print(
        f"BM25 documents    : {len(bm25_documents)}"
    )

    # ---------------------------------------------------------
    # 4. Candidate store
    # ---------------------------------------------------------
    candidates: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------
    # 5. Search every expanded query
    # ---------------------------------------------------------
    for expanded_query in expanded_queries:

        print(
            f"\nSearching expanded query: "
            f"{expanded_query}"
        )

        # =====================================================
        # VECTOR SEARCH
        # =====================================================

        query_embedding = generate_query_embedding(
            expanded_query
        )

        # Same candidate depth as BM25 so neither retriever
        # dominates the fusion; top_k only trims the final list.
        with timed("vector_search"):
            query_result = collection.query(
                query_embeddings=[query_embedding],
                n_results=RETRIEVAL_TOP_K,
                where=metadata_filter if metadata_filter else None,
            )

        result_ids = query_result.get("ids", [[]])[0]
        result_documents = query_result.get(
            "documents",
            [[]],
        )[0]

        result_metadatas = query_result.get(
            "metadatas",
            [[]],
        )[0]

        result_distances = query_result.get(
            "distances",
            [[]],
        )[0]

        for index, chunk_id in enumerate(result_ids):

            vector_rank = index + 1

            distance = result_distances[index]

            # Chroma cosine distance → similarity
            similarity = 1.0 - distance

            # Optional threshold
            if similarity < SIMILARITY_THRESHOLD:
                continue

            if chunk_id not in candidates:

                candidates[chunk_id] = {
                    "id": chunk_id,
                    "text": result_documents[index],
                    "metadata": result_metadatas[index],

                    "vector_score": similarity,
                    "bm25_score": 0.0,

                    "rrf_score": 0.0,

                    "vector_rank_hits": [],
                    "bm25_rank_hits": [],

                    "matched_queries": [],

                    "query_intent": intent,
                    "intent_justification": justification,
                }

            candidate = candidates[chunk_id]

            # Keep best vector similarity
            candidate["vector_score"] = max(
                candidate["vector_score"],
                similarity,
            )

            # Record this query's vector rank
            candidate["vector_rank_hits"].append(
                {
                    "query": expanded_query,
                    "rank": vector_rank,
                }
            )

            # Add RRF contribution
            candidate["rrf_score"] += (
                1.0 / (RRF_K + vector_rank)
            )

            if expanded_query not in candidate["matched_queries"]:
                candidate["matched_queries"].append(
                    expanded_query
                )

        # =====================================================
        # BM25 SEARCH
        # =====================================================

        with timed("bm25"):
            bm25_results = (
                search_bm25_index(
                    bm25=bm25_index,
                    documents=bm25_documents,
                    query=expanded_query,
                    top_k=RETRIEVAL_TOP_K,
                )
                if bm25_index is not None
                else []
            )

        for index, bm25_result in enumerate(
            bm25_results
        ):

            bm25_rank = index + 1

            chunk_id = bm25_result["id"]

            if chunk_id not in candidates:

                candidates[chunk_id] = {
                    "id": chunk_id,
                    "text": bm25_result["text"],
                    "metadata": bm25_result["metadata"],

                    "vector_score": 0.0,
                    "bm25_score": bm25_result["bm25_score"],

                    "rrf_score": 0.0,

                    "vector_rank_hits": [],
                    "bm25_rank_hits": [],

                    "matched_queries": [],

                    "query_intent": intent,
                    "intent_justification": justification,
                }

            candidate = candidates[chunk_id]

            # Keep best BM25 score
            candidate["bm25_score"] = max(
                candidate["bm25_score"],
                bm25_result["bm25_score"],
            )

            # Record this query's BM25 rank
            candidate["bm25_rank_hits"].append(
                {
                    "query": expanded_query,
                    "rank": bm25_rank,
                }
            )

            # Add RRF contribution
            candidate["rrf_score"] += (
                1.0 / (RRF_K + bm25_rank)
            )

            if expanded_query not in candidate["matched_queries"]:
                candidate["matched_queries"].append(
                    expanded_query
                )

    # ---------------------------------------------------------
    # 6. Sort using accumulated RRF score
    # ---------------------------------------------------------
    results = sorted(
        candidates.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    # ---------------------------------------------------------
    # 7. Return top results
    # ---------------------------------------------------------
    final_results = results[:top_k]

    record_retrieval(
        intent=intent,
        justification=justification,
        metadata_filter=metadata_filter,
        expanded_queries=expanded_queries,
        candidate_count=len(candidates),
        final_results=final_results,
    )

    return final_results
    

# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------




def print_results(
    results: list[dict[str, Any]],
    question: str = "",
) -> None:

    print("\n" + "=" * 100)
    print("RETRIEVAL RESULTS")
    print("=" * 100)

    if question:
        print(f"Question          : {question}")

    for index, result in enumerate(results, start=1):

        print(f"\nRank              : {index}")
        print(f"RRF Score         : {result['rrf_score']:.6f}")
        print(f"Vector Score      : {result['vector_score']:.4f}")
        print(f"BM25 Score        : {result['bm25_score']:.4f}")

        print(
            f"Vector Rank Hits  : "
            f"{result['vector_rank_hits']}"
        )

        print(
            f"BM25 Rank Hits    : "
            f"{result['bm25_rank_hits']}"
        )

        print(f"Chunk ID           : {result['id']}")
        print(f"Intent             : {result['query_intent']}")

        print(
            f"Justification      : "
            f"{result['intent_justification']}"
        )

        print(
            f"Matched Queries   : "
            f"{result['matched_queries']}"
        )

        metadata = result.get("metadata", {})

        print(
            f"Top Level         : "
            f"{metadata.get('top_level')}"
        )

        print(
            f"Project           : "
            f"{metadata.get('project')}"
        )

        print(
            f"File              : "
            f"{metadata.get('file_name')}"
        )

        print(
            f"Section           : "
            f"{metadata.get('heading_path')}"
        )

        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")

        if page_start is None:
            pages = "-"
        elif page_end is None or page_end == page_start:
            pages = str(page_start)
        else:
            pages = f"{page_start}-{page_end}"

        print(
            f"Pages             : "
            f"{pages}"
        )

        print("-" * 100)

        print("Document:")
        print(result["text"])

        print("-" * 100)



# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------


def main() -> None:

    print("=" * 80)
    print("TECSACK RAG CHATBOT TEST")
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
        print("Question cannot be empty.")
        return

    # ---------------------------------------------------------
    # 1-4. RETRIEVAL -> CONTEXT -> ANSWER (traced request)
    # ---------------------------------------------------------

    # Imported here: chat_service imports this module.
    from app.chat_service import answer_question
    from app.observability import configure_logging

    configure_logging()

    result = answer_question(
        question=question,
        top_k=TOP_K,
    )

    print_results(
        question=question,
        results=result.results,
    )

    print("\n" + "=" * 100)
    print("LLM CONTEXT")
    print("=" * 100)

    print(result.context)

    print("\n" + "=" * 100)
    print("GENERATED ANSWER")
    print("=" * 100)

    print(result.answer)

    print("=" * 100)
    print(f"trace_id: {result.trace_id}")

if __name__ == "__main__":
    main()
