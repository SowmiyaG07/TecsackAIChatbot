from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator
from urllib.parse import urlparse

from dotenv import load_dotenv

from app.observability.trace import (
    RequestTrace,
    _reset_current_trace,
    _set_current_trace,
    create_trace_id,
    utc_now,
)


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {
        "1", "true", "yes", "on",
    }


def is_observability_enabled() -> bool:
    return _env_flag("OBSERVABILITY_ENABLED", "true")


def is_sql_enabled() -> bool:
    return is_observability_enabled() and _env_flag(
        "OBSERVABILITY_SQL_ENABLED",
        "true",
    )


MAX_ERROR_MESSAGE_CHARS = 4000


# ---------------------------------------------------------
# Structured logging
# ---------------------------------------------------------

logger = logging.getLogger("tecsack.observability")

# Field names whose values must never be written to logs.
_SENSITIVE_KEY = re.compile(
    r"pass(word)?|pwd|secret|api[_-]?key|credential|"
    r"connection[_-]?string|authorization|cookie",
    re.IGNORECASE,
)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Basic console logging for CLI runs. Does nothing if the
    application (e.g. FastAPI/uvicorn) already configured logging.
    """

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

    # One INFO line per Ollama HTTP call drowns the trace events.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _format_value(value: Any) -> str:
    if value is None:
        return "null"

    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, float):
        return f"{value:.1f}"

    if isinstance(value, (int,)):
        return str(value)

    text = str(value)

    if not text or re.search(r"[\s=\"]", text):
        return json.dumps(text, ensure_ascii=False)

    return text


def _connection_string_secrets(connection_string: str) -> list[str]:
    """
    Password values inside an ODBC connection string
    (PWD=...; / Password=...; / PWD={...};).
    """

    return [
        value.strip("{}")
        for value in re.findall(
            r"(?i)(?:^|;)\s*(?:PWD|PASSWORD)\s*=\s*(\{[^}]*\}|[^;]*)",
            connection_string,
        )
        if value.strip("{}")
    ]


def _scrub_secrets(text: str) -> str:
    """
    Remove configured secret values from free text such as
    exception messages.
    """

    connection_string = os.getenv("SQLSERVER_CONNECTION_STRING", "")

    if connection_string:
        text = text.replace(connection_string, "***")

    for secret in _connection_string_secrets(connection_string):
        text = text.replace(secret, "***")

    # redis://user:password@host:port/db
    redis_password = urlparse(os.getenv("REDIS_URL", "")).password

    if redis_password:
        text = text.replace(redis_password, "***")

    return re.sub(
        r"(?i)\b(PWD|PASSWORD)\s*=\s*[^;]*",
        r"\1=***",
        text,
    )


def record_event(
    event: str,
    trace_id: str | None = None,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """
    Write one structured key=value log line:

        trace_id=... event=llm.completed call_purpose=... latency_ms=...

    Never raises.
    """

    if not is_observability_enabled():
        return

    try:
        parts = [
            f"trace_id={trace_id or 'null'}",
            f"event={event}",
        ]

        for key, value in fields.items():

            if _SENSITIVE_KEY.search(key):
                value = "***"

            elif isinstance(value, str):
                value = _scrub_secrets(value)

            parts.append(f"{key}={_format_value(value)}")

        logger.log(level, " ".join(parts))

    except Exception:  # pragma: no cover - logging must never break a request
        pass


def format_error(exc: BaseException, stage: str | None = None) -> str:
    message = _scrub_secrets(f"{type(exc).__name__}: {exc}")

    if stage:
        message = f"[{stage}] {message}"

    return message[:MAX_ERROR_MESSAGE_CHARS]


# ---------------------------------------------------------
# SQL Server store
# ---------------------------------------------------------

_INSERT_CONVERSATION = """
INSERT INTO dbo.ConversationHistory (
    trace_id, session_id, question, answer,
    created_at, completed_at, model_name,
    tool_names, tool_call_count,
    query_intent, expanded_query_count,
    retrieved_chunk_count, final_top_k_count,
    total_latency_ms, retrieval_latency_ms, generation_latency_ms,
    query_expansion_latency_ms, embedding_latency_ms,
    vector_search_latency_ms, bm25_latency_ms,
    retrieval_details, result_type, status, error_message
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_LLM_USAGE = """
INSERT INTO dbo.LLMUsage (
    trace_id, call_purpose, model_name,
    input_tokens, output_tokens, total_tokens,
    latency_ms, estimated_cost, currency, status, created_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Retrievers whose invocations count as "tool calls".
_RETRIEVER_STAGES = ("vector_search", "bm25")


def _sql_datetime(value: datetime | None) -> datetime | None:
    # datetime2 has no offset; all values are stored as UTC.
    return value.replace(tzinfo=None) if value else None


def _truncate(value: str | None, limit: int) -> str | None:
    return value[:limit] if value is not None else None


def build_conversation_row(trace: RequestTrace) -> tuple:
    tools = [
        stage
        for stage in _RETRIEVER_STAGES
        if trace.stage_counts.get(stage)
    ]

    details = {
        "intent_justification": trace.intent_justification,
        "metadata_filter": trace.metadata_filter,
        "expanded_queries": trace.expanded_queries,
        "top_chunks": trace.top_chunks,
        "memory": {
            "status": trace.memory_status,
            "history_messages": trace.history_message_count,
            "history_token_estimate": trace.history_token_estimate,
        },
        "failed_stage": trace.failed_stage,
        "timings_ms": {
            stage: round(ms, 1)
            for stage, ms in trace.timings_ms.items()
        },
        "stage_counts": trace.stage_counts,
    }

    return (
        trace.trace_id,
        _truncate(trace.session_id, 100),
        trace.question,
        trace.answer,
        _sql_datetime(trace.started_at),
        _sql_datetime(trace.completed_at),
        _truncate(trace.model_name, 100),
        _truncate(",".join(tools) or None, 200),
        sum(trace.stage_counts.get(stage, 0) for stage in _RETRIEVER_STAGES),
        _truncate(trace.query_intent, 20),
        len(trace.expanded_queries) if trace.expanded_queries else None,
        trace.retrieved_chunk_count,
        trace.final_top_k_count,
        trace.total_latency_ms,
        trace.timing("retrieval"),
        trace.timing("generation"),
        trace.timing("query_expansion"),
        trace.timing("embedding"),
        trace.timing("vector_search"),
        trace.timing("bm25_index", "bm25"),
        json.dumps(details, ensure_ascii=False, default=str),
        _truncate(trace.result_type, 50),
        _truncate(trace.status, 20),
        trace.error_message,
    )


def build_llm_usage_rows(trace: RequestTrace) -> list[tuple]:
    return [
        (
            trace.trace_id,
            _truncate(call.call_purpose, 50),
            _truncate(call.model_name, 100) or "unknown",
            call.input_tokens,
            call.output_tokens,
            call.total_tokens,
            call.latency_ms,
            call.estimated_cost,
            call.currency,
            _truncate(call.status, 20),
            _sql_datetime(call.created_at),
        )
        for call in trace.llm_calls
    ]


class SqlServerObservabilityStore:
    """
    Writes one ConversationHistory row and its LLMUsage rows in a
    single transaction.

    Configuration (.env), SQL authentication:

        SQLSERVER_CONNECTION_STRING=Driver={ODBC Driver 18 for SQL Server};
            Server=...;Database=TecsackAIChatbot;UID=...;PWD=...;
            Encrypt=yes;TrustServerCertificate=no;
        SQLSERVER_TIMEOUT_SECONDS=5   (optional, login timeout)

    The connection string holds a password: it is never logged.
    """

    def __init__(self, connect=None) -> None:
        # Injectable for tests; defaults to pyodbc.connect.
        self._connect = connect

    @staticmethod
    def is_configured() -> bool:
        return bool(os.getenv("SQLSERVER_CONNECTION_STRING", "").strip())

    def _open(self):
        if self._connect is not None:
            return self._connect()

        import pyodbc  # imported lazily: optional dependency

        return pyodbc.connect(
            os.environ["SQLSERVER_CONNECTION_STRING"],
            timeout=int(os.getenv("SQLSERVER_TIMEOUT_SECONDS", "5")),
            autocommit=False,
        )

    def save_trace(self, trace: RequestTrace) -> None:
        connection = self._open()

        try:
            cursor = connection.cursor()

            # Conversation first: LLMUsage.trace_id references it.
            cursor.execute(
                _INSERT_CONVERSATION,
                build_conversation_row(trace),
            )

            usage_rows = build_llm_usage_rows(trace)

            if usage_rows:
                cursor.executemany(_INSERT_LLM_USAGE, usage_rows)

            connection.commit()

        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass

            raise

        finally:
            connection.close()


_store: SqlServerObservabilityStore | None = None


def get_store() -> SqlServerObservabilityStore:
    global _store

    if _store is None:
        _store = SqlServerObservabilityStore()

    return _store


def set_store(store: SqlServerObservabilityStore | None) -> None:
    """Replace the store (tests, or a custom sink)."""

    global _store
    _store = store


# ---------------------------------------------------------
# Request lifecycle
# ---------------------------------------------------------

def persist_trace(trace: RequestTrace) -> bool:
    """
    Best-effort save to SQL Server. Returns True when saved.
    Failures are logged, never raised.
    """

    if not is_sql_enabled():
        return False

    store = get_store()

    if not store.is_configured() and store._connect is None:
        record_event(
            "observability.persist_skipped",
            trace.trace_id,
            level=logging.WARNING,
            reason="SQLSERVER_CONNECTION_STRING not configured",
        )
        return False

    started = time.perf_counter()

    try:
        store.save_trace(trace)

    except Exception as exc:
        record_event(
            "observability.persist_failed",
            trace.trace_id,
            level=logging.ERROR,
            error=format_error(exc),
        )
        return False

    record_event(
        "observability.persisted",
        trace.trace_id,
        llm_calls=len(trace.llm_calls),
        latency_ms=round((time.perf_counter() - started) * 1000),
    )

    return True


def finish_trace(
    trace: RequestTrace,
    error: BaseException | None = None,
) -> None:
    """
    Close the trace, log the outcome and persist it.
    """

    trace.timings_ms["request"] = (
        time.perf_counter() - trace.start_perf
    ) * 1000
    trace.completed_at = utc_now()

    if error is None:
        trace.status = "success"

        record_event(
            "request.completed",
            trace.trace_id,
            status=trace.status,
            result_type=trace.result_type,
            total_latency_ms=trace.total_latency_ms,
            retrieval_latency_ms=trace.timing("retrieval"),
            generation_latency_ms=trace.timing("generation"),
            llm_calls=len(trace.llm_calls),
            answer_chars=len(trace.answer or ""),
        )

    else:
        trace.status = "error"
        trace.result_type = trace.result_type or "error"
        trace.error_message = format_error(error, trace.failed_stage)

        record_event(
            "request.failed",
            trace.trace_id,
            level=logging.ERROR,
            status=trace.status,
            failed_stage=trace.failed_stage,
            error_type=type(error).__name__,
            total_latency_ms=trace.total_latency_ms,
        )

    persist_trace(trace)


@contextmanager
def start_trace(
    question: str,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> Iterator[RequestTrace]:
    """
    Open a trace for one chatbot request:

        with start_trace(question, session_id) as trace:
            ...  # retrieval, generation

    The application's own exception always propagates unchanged;
    observability failures are logged and swallowed.
    """

    trace = RequestTrace(
        trace_id=trace_id or create_trace_id(),
        question=question,
        session_id=session_id,
    )

    token = _set_current_trace(trace)

    record_event(
        "request.started",
        trace.trace_id,
        session_id=session_id,
        question_chars=len(question),
    )

    try:
        yield trace

    except BaseException as exc:
        _finish_safely(trace, exc)
        raise

    else:
        _finish_safely(trace, None)

    finally:
        _reset_current_trace(token)


def _finish_safely(
    trace: RequestTrace,
    error: BaseException | None,
) -> None:
    try:
        finish_trace(trace, error)

    except Exception as exc:  # observability must never mask the request result
        record_event(
            "observability.internal_error",
            trace.trace_id,
            level=logging.ERROR,
            error=format_error(exc),
        )
