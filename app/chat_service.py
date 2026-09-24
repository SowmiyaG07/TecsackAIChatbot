from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import redis

from app.generation.answer_generator import (
    OLLAMA_MODEL,
    generate_answer,
)
from app.generation.context_builder import build_context
from app.memory import RedisConversationMemory, estimate_history_tokens
from app.observability import (
    RequestTrace,
    format_error,
    record_event,
    start_trace,
    timed,
)


# Fixed reply from answer_generator when the context has no answer.
NOT_FOUND_PREFIX = "I couldn't find that information"


@dataclass
class ChatResult:
    trace_id: str
    answer: str
    results: list[dict[str, Any]]
    context: str
    result_type: str
    history: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------
# Conversation memory (Redis, best-effort)
# ---------------------------------------------------------
#
# Redis only adds conversational context. Every failure here is
# logged and the request continues without history.

_default_memory: RedisConversationMemory | None = None


def get_memory() -> RedisConversationMemory | None:
    """
    Shared memory instance (one Redis connection pool per process).
    None when Redis is not configured.
    """

    global _default_memory

    if _default_memory is None:
        try:
            _default_memory = RedisConversationMemory()
        except Exception as exc:
            # Creating the client does not connect, so this is a
            # configuration problem (REDIS_URL, history limits).
            record_event(
                "redis.config.invalid",
                error=format_error(exc),
            )
            return None

    return _default_memory


def _is_connection_error(exc: Exception) -> bool:
    return isinstance(exc, (redis.ConnectionError, redis.TimeoutError))


def _record_memory_failure(
    trace: RequestTrace,
    session_id: str,
    operation: str,
    exc: Exception,
) -> None:
    record_event(
        "redis.connection.failed"
        if _is_connection_error(exc)
        else f"redis.{operation}.failed",
        trace.trace_id,
        session_id=session_id,
        operation=operation,
        error_type=type(exc).__name__,
        error=format_error(exc),
    )


def load_history(
    memory: RedisConversationMemory | None,
    session_id: str | None,
    trace: RequestTrace,
) -> list[dict[str, str]] | None:
    """
    Recent messages for the LLM (oldest first).

    Returns [] for no/empty session and None when Redis failed,
    so the caller can skip saving to an unavailable Redis.
    """

    if not session_id:
        trace.memory_status = "no_session"
        return []

    if memory is None:
        trace.memory_status = "unavailable"
        return None

    error: Exception | None = None

    # The try sits inside timed() so a Redis failure is not
    # reported as the request's failed stage.
    with timed("redis_history_load"):
        try:
            history = memory.get_messages_for_llm(session_id)
        except Exception as exc:
            error = exc

    if error is not None:
        trace.memory_status = "unavailable"
        _record_memory_failure(trace, session_id, "history_load", error)
        return None

    trace.memory_status = "loaded" if history else "empty"
    trace.history_message_count = len(history)
    trace.history_token_estimate = estimate_history_tokens(history)

    record_event(
        "redis.history.loaded",
        trace.trace_id,
        session_id=session_id,
        message_count=trace.history_message_count,
        history_token_estimate=trace.history_token_estimate,
        latency_ms=round(trace.timings_ms.get("redis_history_load", 0)),
    )

    return history


def save_turn(
    memory: RedisConversationMemory,
    session_id: str,
    question: str,
    answer: str,
    trace: RequestTrace,
) -> None:
    """
    Store the question and its answer as one atomic write.
    Failures are logged; the answer is returned regardless.
    """

    error: Exception | None = None

    with timed("redis_history_save"):
        try:
            stored = memory.add_messages(
                session_id,
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            )
        except Exception as exc:
            error = exc

    if error is not None:
        _record_memory_failure(trace, session_id, "message_save", error)
        return

    record_event(
        "redis.message.saved",
        trace.trace_id,
        session_id=session_id,
        messages_saved=2,
        message_count=stored,
        latency_ms=round(trace.timings_ms.get("redis_history_save", 0)),
    )


def clear_session_history(
    session_id: str,
    memory: RedisConversationMemory | None = None,
) -> bool:
    """
    Remove one session's short-term history (SQL Server keeps the
    permanent record). Raises if Redis is unavailable.
    """

    memory = memory or get_memory()

    if memory is None:
        raise RuntimeError("Redis conversation memory is not configured.")

    cleared = memory.clear_history(session_id)

    record_event(
        "redis.history.cleared",
        session_id=session_id,
        had_history=cleared,
    )

    return cleared


def classify_result(
    results: list[dict[str, Any]],
    answer: str,
) -> str:
    if not results:
        return "no_results"

    if answer.strip().startswith(NOT_FOUND_PREFIX):
        return "not_found"

    return "answered"


def answer_question(
    question: str,
    session_id: str | None = None,
    top_k: int | None = None,
    retriever: Callable[..., list[dict[str, Any]]] | None = None,
    memory: RedisConversationMemory | None = None,
    trace_id: str | None = None,
) -> ChatResult:
    """
    One chatbot request, traced end to end:

        trace_id -> Redis history (session_id)
                 -> intent -> expansion -> vector + BM25 -> RRF
                 -> context -> answer generation (history + documents)
                 -> Redis: save question + answer
                 -> SQL Server: ConversationHistory + LLMUsage

    Retrieval uses only the current question; history is given to
    the answer generator to resolve references ("what about the
    CTO?"), never as a source of facts.

    Without a session_id no history is read or written.

    `retriever` defaults to vector_retrieval.search_vectors and
    `memory` to the shared Redis memory; both are parameters so
    tests can run without ChromaDB or Redis.

    `trace_id` lets a caller (the API) choose the id up front, so it
    can be reported even when the request fails. Generated if omitted.
    """

    if retriever is None:
        # Imported lazily: loading it opens the Chroma collection.
        from app.retrieval.vector_retrieval import TOP_K, search_vectors

        retriever = search_vectors
        top_k = top_k or TOP_K

    elif top_k is None:
        top_k = int(os.getenv("TOP_K", "5"))

    with start_trace(
        question,
        session_id=session_id,
        trace_id=trace_id,
    ) as trace:

        trace.model_name = OLLAMA_MODEL

        if session_id and memory is None:
            memory = get_memory()

        # Read BEFORE saving this turn, so the current question is
        # not duplicated in the history given to the LLM.
        history = load_history(memory, session_id, trace)

        with timed("retrieval", emit_event=True) as span:
            results = retriever(
                question=question,
                top_k=top_k,
            )

            span["chunks"] = len(results)

        with timed("context_build"):
            context = build_context(results)

        with timed("generation", emit_event=True):
            answer = generate_answer(
                question=question,
                context=context,
                history=history or None,
            )

        trace.answer = answer
        trace.result_type = classify_result(results, answer)

        # history is None when Redis already failed in this request:
        # skip the save instead of waiting for a second timeout.
        if session_id and memory is not None and history is not None:
            save_turn(memory, session_id, question, answer, trace)

        return ChatResult(
            trace_id=trace.trace_id,
            answer=answer,
            results=results,
            context=context,
            result_type=trace.result_type,
            history=history or [],
        )


# ---------------------------------------------------------
# Manual test
# ---------------------------------------------------------
#
#   One question, no memory:
#     python -m app.chat_service "who is the CEO of tecsack"
#
#   Conversation with Redis memory:
#     python -m app.chat_service --session test-session-001 "Who is the CEO of Tecsack?"
#     python -m app.chat_service --session test-session-001 "What about the CTO?"
#
#   Interactive conversation (empty line to quit):
#     python -m app.chat_service --session test-session-001
#
#   Clear a session's short-term history:
#     python -m app.chat_service --session test-session-001 --clear

def _print_result(result: ChatResult) -> None:
    print()
    print("=" * 80)
    print(f"trace_id        : {result.trace_id}")
    print(f"result_type     : {result.result_type}")
    print(f"history to LLM  : {len(result.history)} message(s)")

    for message in result.history:
        print(f"  {message['role']:>9}: {message['content'][:70]}")

    print("=" * 80)
    print(result.answer)
    print()


def main() -> None:
    from app.observability import configure_logging

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Ask the Tecsack RAG chatbot.",
    )
    parser.add_argument("question", nargs="*", help="question to ask")
    parser.add_argument(
        "--session",
        help="session id; enables Redis conversation memory",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="clear the session's Redis history and exit",
    )
    args = parser.parse_args()

    configure_logging()

    if args.clear:
        if not args.session:
            parser.error("--clear requires --session")

        cleared = clear_session_history(args.session)
        print(f"Cleared {args.session}: {cleared}")
        return

    question = " ".join(args.question).strip()

    if question:
        _print_result(answer_question(question, session_id=args.session))
        return

    # Interactive conversation.
    while True:
        question = input("You: ").strip()

        if not question:
            return

        _print_result(answer_question(question, session_id=args.session))


if __name__ == "__main__":
    main()
