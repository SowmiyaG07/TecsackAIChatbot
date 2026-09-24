from __future__ import annotations

from typing import Any

# Reuse the project's existing token counter (tiktoken cl100k_base,
# with a ~4 characters/token fallback) instead of adding another one.
from app.ingestion.chunker import count_tokens


# Chat templates wrap every message in role/header tokens
# (llama3: <|start_header_id|>role<|end_header_id|> ... <|eot_id|>).
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_message_tokens(
    message: dict[str, Any],
) -> int:
    """
    Estimated prompt tokens for one chat message.

    This is an ESTIMATE for keeping the history within its prompt
    budget: cl100k_base is close to, but not identical to, the
    llama3 tokenizer. Billing/usage numbers come from Ollama's
    reported counts (see app.observability), never from here.
    """

    return (
        count_tokens(str(message.get("content", "")))
        + MESSAGE_OVERHEAD_TOKENS
    )


__all__ = [
    "MESSAGE_OVERHEAD_TOKENS",
    "count_tokens",
    "estimate_message_tokens",
]
