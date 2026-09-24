from __future__ import annotations

import json
import os

import redis
from dotenv import load_dotenv

from app.memory.redis_client import create_redis_client
from app.memory.token_utils import estimate_message_tokens


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

KEY_PREFIX = "chat:history:"

VALID_ROLES = {"user", "assistant"}

MAX_SESSION_ID_LENGTH = 128


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


# ---------------------------------------------------------
# History selection (pure, no Redis)
# ---------------------------------------------------------

def select_recent_messages(
    messages: list[dict[str, str]],
    max_messages: int,
    max_tokens: int,
) -> list[dict[str, str]]:
    """
    Pick the most recent messages that fit both limits.

    1. Consider only the newest `max_messages` messages.
    2. Walk from the newest message backwards, adding messages
       while the estimated token total stays within `max_tokens`.
    3. Stop at the first message that does not fit, so the history
       never has gaps.
    4. Return the selection in chronological order (oldest first).

    A single newest message larger than the budget yields [].
    """

    if max_messages <= 0 or max_tokens <= 0:
        return []

    selected: list[dict[str, str]] = []
    used_tokens = 0

    for message in reversed(messages[-max_messages:]):

        message_tokens = estimate_message_tokens(message)

        if used_tokens + message_tokens > max_tokens:
            break

        selected.append(message)
        used_tokens += message_tokens

    selected.reverse()

    return selected


def estimate_history_tokens(
    messages: list[dict[str, str]],
) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


# ---------------------------------------------------------
# Redis memory
# ---------------------------------------------------------

class RedisConversationMemory:
    """
    Short-term conversation memory, one Redis list per session:

        chat:history:{session_id}  ->  [ '{"role": "user", "content": "..."}', ... ]

    Messages use the Ollama chat format ({"role", "content"}).
    Only the newest MAX_HISTORY_MESSAGES are kept, and each session
    key expires HISTORY_TTL_SECONDS after its last write
    (HISTORY_TTL_SECONDS <= 0 disables expiry).

    Redis is not the source of truth: permanent history lives in
    SQL Server (ConversationHistory). Methods raise redis.RedisError
    on connection problems (except ping); callers decide whether
    that is fatal. The chatbot treats it as non-fatal.

    .env:
        MAX_HISTORY_MESSAGES=7
        HISTORY_MAX_TOKENS=1500
        HISTORY_TTL_SECONDS=86400
    """

    def __init__(
        self,
        client: redis.Redis | None = None,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        ttl_seconds: int | None = None,
    ) -> None:

        self.client = (
            client
            if client is not None
            else create_redis_client()
        )

        self.max_messages = (
            max_messages
            if max_messages is not None
            else _env_int("MAX_HISTORY_MESSAGES", "7")
        )

        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else _env_int("HISTORY_MAX_TOKENS", "1500")
        )

        self.ttl_seconds = (
            ttl_seconds
            if ttl_seconds is not None
            else _env_int("HISTORY_TTL_SECONDS", "86400")
        )

        # LTRIM/LRANGE with -0 means "everything": a limit of 0 would
        # silently disable trimming, so it is rejected.
        if self.max_messages < 1:
            raise ValueError("MAX_HISTORY_MESSAGES must be at least 1.")

        if self.max_tokens < 1:
            raise ValueError("HISTORY_MAX_TOKENS must be at least 1.")

    # -----------------------------------------------------
    # Keys and validation
    # -----------------------------------------------------

    @staticmethod
    def key(session_id: str) -> str:
        session_id = (session_id or "").strip()

        if not session_id:
            raise ValueError("session_id cannot be empty.")

        if len(session_id) > MAX_SESSION_ID_LENGTH:
            raise ValueError(
                f"session_id cannot exceed "
                f"{MAX_SESSION_ID_LENGTH} characters."
            )

        return f"{KEY_PREFIX}{session_id}"

    @staticmethod
    def _serialize(role: str, content: str) -> str:
        if role not in VALID_ROLES:
            raise ValueError(
                f"Unsupported role {role!r}; "
                f"expected one of {sorted(VALID_ROLES)}."
            )

        content = (content or "").strip()

        if not content:
            raise ValueError("Message content cannot be empty.")

        return json.dumps(
            {"role": role, "content": content},
            ensure_ascii=False,
        )

    @staticmethod
    def _deserialize(raw: str) -> dict[str, str] | None:
        try:
            message = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None

        if (
            not isinstance(message, dict)
            or message.get("role") not in VALID_ROLES
            or not isinstance(message.get("content"), str)
        ):
            return None

        return {
            "role": message["role"],
            "content": message["content"],
        }

    # -----------------------------------------------------
    # Operations
    # -----------------------------------------------------

    def ping(self) -> bool:
        """True when Redis is reachable. Never raises."""

        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> int:
        """
        Append one message. Returns the number of stored messages.
        """

        return self.add_messages(
            session_id,
            [{"role": role, "content": content}],
        )

    def add_messages(
        self,
        session_id: str,
        messages: list[dict[str, str]],
    ) -> int:
        """
        Append several messages atomically (e.g. a user question and
        its answer), trim to the newest max_messages and refresh the
        TTL. Returns the number of stored messages.
        """

        key = self.key(session_id)

        payload = [
            self._serialize(message.get("role", ""), message.get("content", ""))
            for message in messages
        ]

        if not payload:
            return self.get_message_count(session_id)

        pipeline = self.client.pipeline(transaction=True)

        pipeline.rpush(key, *payload)
        pipeline.ltrim(key, -self.max_messages, -1)

        if self.ttl_seconds > 0:
            pipeline.expire(key, self.ttl_seconds)

        pipeline.llen(key)

        return int(pipeline.execute()[-1])

    def get_messages(
        self,
        session_id: str,
    ) -> list[dict[str, str]]:
        """
        Stored messages for the session, oldest first.
        An unknown session returns [].
        """

        raw_messages = self.client.lrange(
            self.key(session_id),
            -self.max_messages,
            -1,
        )

        messages = [
            self._deserialize(raw)
            for raw in raw_messages
        ]

        # Skip corrupted entries rather than failing the request.
        return [message for message in messages if message]

    def get_messages_for_llm(
        self,
        session_id: str,
    ) -> list[dict[str, str]]:
        """
        Recent messages within MAX_HISTORY_MESSAGES and
        HISTORY_MAX_TOKENS, oldest first, in Ollama chat format.
        """

        return select_recent_messages(
            self.get_messages(session_id),
            max_messages=self.max_messages,
            max_tokens=self.max_tokens,
        )

    def clear_history(
        self,
        session_id: str,
    ) -> bool:
        """
        Delete this session's history only.
        Returns True if there was history to delete.
        """

        return bool(self.client.delete(self.key(session_id)))

    def get_message_count(
        self,
        session_id: str,
    ) -> int:
        return int(self.client.llen(self.key(session_id)))
