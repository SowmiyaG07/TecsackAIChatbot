from __future__ import annotations

import json
import re
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def load_documents(
    collection,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    kwargs = {}

    if metadata_filter:
        kwargs["where"] = metadata_filter

    data = collection.get(**kwargs)

    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    results = []

    for index, chunk_id in enumerate(ids):
        results.append(
            {
                "id": chunk_id,
                "text": documents[index],
                "metadata": metadatas[index],
            }
        )

    return results


def build_bm25_index(
    documents: list[dict[str, Any]],
) -> BM25Okapi:
    tokenized_documents = [
        tokenize(document["text"])
        for document in documents
    ]

    return BM25Okapi(tokenized_documents)


# ---------------------------------------------------------
# Index cache
# ---------------------------------------------------------
#
# Loading every chunk from Chroma and re-tokenizing it on every
# question is wasteful. Indexes are cached per metadata filter and
# rebuilt only when the collection's size changes (i.e. after an
# ingestion run). Call clear_bm25_cache() after in-process updates
# that keep the same chunk count.

_BM25_CACHE: dict[
    str,
    tuple[int, BM25Okapi, list[dict[str, Any]]],
] = {}


def clear_bm25_cache() -> None:
    _BM25_CACHE.clear()


def get_bm25_index(
    collection,
    metadata_filter: dict[str, Any] | None = None,
) -> tuple[BM25Okapi | None, list[dict[str, Any]]]:
    """
    Return (bm25_index, documents) for the given filter.

    bm25_index is None when no documents match the filter.
    """

    cache_key = json.dumps(
        metadata_filter or {},
        sort_keys=True,
    )

    collection_size = collection.count()

    cached = _BM25_CACHE.get(cache_key)

    if cached and cached[0] == collection_size:
        return cached[1], cached[2]

    documents = load_documents(
        collection,
        metadata_filter=metadata_filter,
    )

    # BM25Okapi fails on an empty corpus.
    if not documents:
        return None, []

    bm25 = build_bm25_index(documents)

    _BM25_CACHE[cache_key] = (
        collection_size,
        bm25,
        documents,
    )

    return bm25, documents


def search_bm25_index(
    bm25: BM25Okapi,
    documents: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indexes[:top_k]:

        # Zero score = no query term matched; not a real hit.
        if scores[index] <= 0:
            break

        results.append(
            {
                "id": documents[index]["id"],
                "text": documents[index]["text"],
                "metadata": documents[index]["metadata"],
                "bm25_score": float(scores[index]),
            }
        )

    return results
