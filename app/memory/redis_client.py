from __future__ import annotations

import os

import redis
from dotenv import load_dotenv


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Client factory
# ---------------------------------------------------------

def create_redis_client(
    url: str | None = None,
) -> redis.Redis:
    """
    Create a Redis client from configuration.

    .env:
        REDIS_URL=redis://localhost:6379/0
        REDIS_SOCKET_TIMEOUT_SECONDS=2   (optional)

    The timeouts are short on purpose: Redis only holds short-term
    conversation memory, so an unavailable Redis must fail fast
    instead of stalling the chatbot request.

    Creating the client does not connect; the first command does.
    """

    redis_url = url or os.getenv("REDIS_URL")

    if not redis_url:
        raise ValueError("REDIS_URL is not configured.")

    timeout = float(
        os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "2")
    )

    return redis.Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=timeout,
        socket_timeout=timeout,
    )
