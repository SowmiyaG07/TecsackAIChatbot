"""
Observability tests.

No Ollama, ChromaDB or SQL Server needed: the LLM client, the
retriever and the SQL store are replaced with fakes. The real
expand_query / generate_answer / answer_question code paths run.

    venv\\Scripts\\python -m pytest tests/test_observability.py -v
"""

from __future__ import annotations

import json
import logging
import uuid

import pytest

import app.generation.answer_generator as answer_generator
import app.retrieval.query_expansion as query_expansion
from app.chat_service import answer_question
from app.observability import (
    SqlServerObservabilityStore,
    extract_token_usage,
    record_event,
    record_retrieval,
    set_store,
    timed,
)
from app.observability.monitoring import (
    build_conversation_row,
    build_llm_usage_rows,
)

SECRET = "Sup3r-S3cret-Pa55!"


# ---------------------------------------------------------
# Fakes
# ---------------------------------------------------------

class FakeOllamaClient:
    """
    Returns Ollama-shaped chat responses. Query expansion and answer
    generation are told apart by the system prompt.
    """

    def __init__(
        self,
        expansion_usage=(120, 30),
        answer_usage=(900, 45),
        fail_on=None,
    ):
        self.expansion_usage = expansion_usage
        self.answer_usage = answer_usage
        self.fail_on = fail_on
        self.calls = []

    def chat(self, **kwargs):
        is_answer = kwargs["messages"][0]["role"] == "system"
        purpose = "answer_generation" if is_answer else "query_expansion"
        self.calls.append(purpose)

        if self.fail_on == purpose:
            raise RuntimeError(f"{purpose} model crashed")

        if is_answer:
            content = "The CEO is Sakthivel (About US - Google Docs.pdf, page 6)."
            usage = self.answer_usage
        else:
            content = json.dumps({"queries": ["leadership team", "CEO"]})
            usage = self.expansion_usage

        response = {
            "model": kwargs["model"],
            "message": {"role": "assistant", "content": content},
        }

        if usage is not None:
            response["prompt_eval_count"] = usage[0]
            response["eval_count"] = usage[1]

        return response


RESULTS = [
    {
        "id": "doc_company_about_us_google_docs_chunk_00014",
        "text": "Chief Executive Officer (CEO) - Sakthivel",
        "metadata": {
            "file_name": "About US - Google Docs.pdf",
            "page_start": 6,
            "page_end": 6,
            "heading_path": "ABOUT US > 4. Leadership Team",
        },
        "rrf_score": 0.13,
    },
    {
        "id": "doc_company_about_us_google_docs_chunk_00015",
        "text": "Team Structure & Talent",
        "metadata": {"file_name": "About US - Google Docs.pdf"},
        "rrf_score": 0.06,
    },
]


def fake_retriever(question, top_k):
    """
    Mirrors search_vectors' instrumentation: the real expand_query
    (LLM call) plus embedding / vector / BM25 stage timers.
    """

    with timed("query_expansion", emit_event=True):
        expanded = query_expansion.expand_query(question, intent="company")

    for _ in expanded:
        with timed("embedding"):
            pass
        with timed("vector_search"):
            pass
        with timed("bm25"):
            pass

    record_retrieval(
        intent="company",
        justification="test",
        metadata_filter={"top_level": "company"},
        expanded_queries=expanded,
        candidate_count=7,
        final_results=RESULTS[:top_k],
    )

    return RESULTS[:top_k]


class FakeStore:
    """Captures traces instead of writing to SQL Server."""

    _connect = None

    def __init__(self, fail_with: Exception | None = None):
        self.saved = []
        self.fail_with = fail_with

    def is_configured(self):
        return True

    def save_trace(self, trace):
        if self.fail_with:
            raise self.fail_with
        self.saved.append(trace)


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture(autouse=True)
def no_redis(monkeypatch):
    """Keep these tests off the real Redis (see test_redis_memory.py)."""
    import app.chat_service as chat_service

    monkeypatch.setattr(chat_service, "get_memory", lambda: None)


@pytest.fixture
def llm(monkeypatch):
    client = FakeOllamaClient()
    monkeypatch.setattr(query_expansion, "ollama_client", client)
    monkeypatch.setattr(answer_generator, "ollama_client", client)
    return client


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("OBSERVABILITY_SQL_ENABLED", "true")
    monkeypatch.setenv(
        "SQLSERVER_CONNECTION_STRING",
        "Driver={ODBC Driver 18 for SQL Server};Server=db;"
        f"Database=TecsackAIChatbot;UID=tecsack_chatbot_app;PWD={SECRET};",
    )

    fake = FakeStore()
    set_store(fake)
    yield fake
    set_store(None)


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="tecsack.observability")
    return caplog


def ask(question="who is the CEO of tecsack"):
    return answer_question(
        question,
        session_id="session-1",
        top_k=5,
        retriever=fake_retriever,
    )


# ---------------------------------------------------------
# 1-2. Trace id
# ---------------------------------------------------------

def test_trace_id_is_generated(llm, store):
    result = ask()

    assert uuid.UUID(result.trace_id).version == 4


def test_same_trace_id_used_throughout_request(llm, store, logs):
    result = ask()

    trace = store.saved[0]
    assert trace.trace_id == result.trace_id

    conversation_row = build_conversation_row(trace)
    assert conversation_row[0] == result.trace_id

    usage_rows = build_llm_usage_rows(trace)
    assert {row[0] for row in usage_rows} == {result.trace_id}

    events = [r.getMessage() for r in logs.records]
    for event in (
        "request.started",
        "query_expansion.completed",
        "retrieval.completed",
        "llm.completed",
        "generation.completed",
        "request.completed",
    ):
        matching = [m for m in events if f"event={event}" in m]
        assert matching, f"missing {event}"
        assert all(f"trace_id={result.trace_id}" in m for m in matching)


def test_each_request_gets_its_own_trace_id(llm, store):
    first = ask()
    second = ask()

    assert first.trace_id != second.trace_id


# ---------------------------------------------------------
# 3-5. LLM usage
# ---------------------------------------------------------

def test_query_expansion_and_answer_generation_usage_recorded(llm, store):
    ask()

    calls = store.saved[0].llm_calls
    assert [c.call_purpose for c in calls] == [
        "query_expansion",
        "answer_generation",
    ]
    assert all(c.model_name == answer_generator.OLLAMA_MODEL for c in calls)
    assert all(c.latency_ms >= 0 for c in calls)


def test_actual_token_counts_recorded(llm, store):
    ask()

    expansion, generation = store.saved[0].llm_calls

    assert (expansion.input_tokens, expansion.output_tokens,
            expansion.total_tokens) == (120, 30, 150)
    assert (generation.input_tokens, generation.output_tokens,
            generation.total_tokens) == (900, 45, 945)


def test_missing_token_counts_stay_null(monkeypatch, store, logs):
    client = FakeOllamaClient(expansion_usage=None, answer_usage=None)
    monkeypatch.setattr(query_expansion, "ollama_client", client)
    monkeypatch.setattr(answer_generator, "ollama_client", client)

    ask()

    for call in store.saved[0].llm_calls:
        assert call.input_tokens is None
        assert call.output_tokens is None
        assert call.total_tokens is None

    assert "token_usage=unavailable" in logs.text


def test_extract_token_usage_partial_values():
    assert extract_token_usage({"eval_count": 10}) == (None, 10, None)

    class Response:  # attribute access, like ollama.ChatResponse
        prompt_eval_count = 5
        eval_count = 7

    assert extract_token_usage(Response()) == (5, 7, 12)


# ---------------------------------------------------------
# 6-9. Latency and ConversationHistory
# ---------------------------------------------------------

def test_latencies_recorded(llm, store):
    ask()

    trace = store.saved[0]

    retrieval = trace.timing("retrieval")
    generation = trace.timing("generation")
    total = trace.total_latency_ms

    assert retrieval is not None
    assert generation is not None
    assert total is not None
    assert total >= retrieval + generation - 1  # rounding

    for stage in ("query_expansion", "embedding", "vector_search", "bm25"):
        assert stage in trace.timings_ms

    # One embedding / vector / BM25 run per expanded query.
    assert trace.stage_counts["embedding"] == len(trace.expanded_queries)


def test_conversation_history_saved_after_success(llm, store):
    result = ask()

    assert len(store.saved) == 1
    trace = store.saved[0]

    assert trace.status == "success"
    assert trace.answer == result.answer
    assert trace.result_type == "answered"
    assert trace.session_id == "session-1"
    assert trace.completed_at is not None

    row = build_conversation_row(trace)
    columns = dict(zip(
        [
            "trace_id", "session_id", "question", "answer",
            "created_at", "completed_at", "model_name",
            "tool_names", "tool_call_count",
            "query_intent", "expanded_query_count",
            "retrieved_chunk_count", "final_top_k_count",
            "total_latency_ms", "retrieval_latency_ms",
            "generation_latency_ms", "query_expansion_latency_ms",
            "embedding_latency_ms", "vector_search_latency_ms",
            "bm25_latency_ms", "retrieval_details", "result_type",
            "status", "error_message",
        ],
        row,
    ))

    assert columns["query_intent"] == "company"
    assert columns["expanded_query_count"] == 3  # original + 2
    assert columns["retrieved_chunk_count"] == 7
    assert columns["final_top_k_count"] == 2
    assert columns["tool_names"] == "vector_search,bm25"
    assert columns["tool_call_count"] == 6
    assert columns["created_at"].tzinfo is None  # stored as UTC datetime2

    details = json.loads(columns["retrieval_details"])
    assert details["expanded_queries"][0] == "who is the CEO of tecsack"
    assert details["top_chunks"][0]["id"].endswith("chunk_00014")


# ---------------------------------------------------------
# 10. Errors
# ---------------------------------------------------------

def test_generation_error_recorded_and_original_error_raised(
    monkeypatch, store, logs,
):
    client = FakeOllamaClient(fail_on="answer_generation")
    monkeypatch.setattr(query_expansion, "ollama_client", client)
    monkeypatch.setattr(answer_generator, "ollama_client", client)

    with pytest.raises(RuntimeError, match="answer_generation model crashed"):
        ask()

    trace = store.saved[0]
    assert trace.status == "error"
    assert trace.failed_stage == "answer_generation"
    assert trace.error_message.startswith(
        "[answer_generation] RuntimeError: answer_generation model crashed"
    )
    assert trace.answer is None

    failed_call = trace.llm_calls[-1]
    assert failed_call.call_purpose == "answer_generation"
    assert failed_call.status == "error"
    assert failed_call.input_tokens is None

    assert "event=request.failed" in logs.text


def test_query_expansion_error_recorded(monkeypatch, store):
    client = FakeOllamaClient(fail_on="query_expansion")
    monkeypatch.setattr(query_expansion, "ollama_client", client)
    monkeypatch.setattr(answer_generator, "ollama_client", client)

    with pytest.raises(RuntimeError):
        ask()

    trace = store.saved[0]
    assert trace.status == "error"
    assert trace.failed_stage == "query_expansion"
    assert trace.timing("generation") is None
    assert client.calls == ["query_expansion"]


def test_retriever_error_recorded(llm, store):
    def broken_retriever(question, top_k):
        with timed("vector_search"):
            raise ConnectionError("chroma unavailable")

    with pytest.raises(ConnectionError):
        answer_question("q", top_k=5, retriever=broken_retriever)

    assert store.saved[0].failed_stage == "vector_search"


# ---------------------------------------------------------
# 11. Observability failures never break the chatbot
# ---------------------------------------------------------

def test_database_failure_does_not_break_answer(llm, store, logs):
    store.fail_with = ConnectionError("SQL Server unreachable")

    result = ask()

    assert result.answer.startswith("The CEO is Sakthivel")
    assert "event=observability.persist_failed" in logs.text


def test_database_failure_does_not_mask_application_error(
    monkeypatch, store,
):
    store.fail_with = ConnectionError("SQL Server unreachable")
    client = FakeOllamaClient(fail_on="answer_generation")
    monkeypatch.setattr(query_expansion, "ollama_client", client)
    monkeypatch.setattr(answer_generator, "ollama_client", client)

    # The LLM error surfaces, not the database error.
    with pytest.raises(RuntimeError, match="model crashed"):
        ask()


def test_internal_observability_bug_does_not_break_answer(
    llm, store, monkeypatch, logs,
):
    import app.observability.monitoring as monitoring

    def broken_finish(trace, error=None):
        raise ValueError("bug in observability")

    monkeypatch.setattr(monitoring, "finish_trace", broken_finish)

    result = ask()

    assert result.answer
    assert "event=observability.internal_error" in logs.text


def test_disabled_observability_still_answers(llm, store, monkeypatch, logs):
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")

    result = ask()

    assert result.answer
    assert store.saved == []
    assert logs.text == ""


# ---------------------------------------------------------
# 12. No secrets in logs
# ---------------------------------------------------------

def test_secrets_never_logged(llm, store, logs):
    # Password both inside a connection string and on its own.
    store.fail_with = Exception(
        f"Login failed: DRIVER={{x}};SERVER=db;UID=sa;PWD={SECRET}; "
        f"(tried password {SECRET})"
    )

    ask()
    record_event("custom.event", password=SECRET, api_key="k-123")

    assert SECRET not in logs.text
    assert "k-123" not in logs.text
    assert "password=***" in logs.text


def test_question_and_answer_text_not_logged(llm, store, logs):
    ask("who is the CEO of tecsack")

    assert "who is the CEO" not in logs.text
    assert "Sakthivel" not in logs.text
    assert "question_chars=" in logs.text


def test_connection_string_password_extraction():
    from app.observability.monitoring import _connection_string_secrets

    assert _connection_string_secrets(
        "Server=x;UID=u;PWD=abc123;Encrypt=yes;"
    ) == ["abc123"]
    assert _connection_string_secrets(
        "Server=x;UID=u;Password={a;b=c};"
    ) == ["a;b=c"]
    assert _connection_string_secrets("Server=x;Trusted_Connection=yes;") == []


def test_sql_not_configured_is_skipped_without_error(llm, monkeypatch, logs):
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("OBSERVABILITY_SQL_ENABLED", "true")
    monkeypatch.setenv("SQLSERVER_CONNECTION_STRING", "")
    set_store(SqlServerObservabilityStore())

    try:
        result = ask()
    finally:
        set_store(None)

    assert result.answer
    assert "event=observability.persist_skipped" in logs.text


# ---------------------------------------------------------
# SQL store (fake pyodbc connection)
# ---------------------------------------------------------

class FakeCursor:
    def __init__(self, log, fail=False):
        self.log = log
        self.fail = fail

    def execute(self, sql, params):
        if self.fail:
            raise RuntimeError("insert failed")
        self.log.append(("execute", sql.split("(")[0].strip(), params))

    def executemany(self, sql, rows):
        self.log.append(("executemany", sql.split("(")[0].strip(), rows))


class FakeConnection:
    def __init__(self, fail=False):
        self.log = []
        self.fail = fail
        self.committed = self.rolled_back = self.closed = False

    def cursor(self):
        return FakeCursor(self.log, self.fail)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_sql_store_writes_conversation_then_usage_in_one_transaction(
    llm, store,
):
    ask()
    trace = store.saved[0]

    connection = FakeConnection()
    SqlServerObservabilityStore(connect=lambda: connection).save_trace(trace)

    assert [entry[1] for entry in connection.log] == [
        "INSERT INTO dbo.ConversationHistory",
        "INSERT INTO dbo.LLMUsage",
    ]
    assert len(connection.log[1][2]) == 2
    assert connection.committed and connection.closed


def test_sql_store_rolls_back_on_failure(llm, store):
    ask()

    connection = FakeConnection(fail=True)

    with pytest.raises(RuntimeError):
        SqlServerObservabilityStore(
            connect=lambda: connection,
        ).save_trace(store.saved[0])

    assert connection.rolled_back and connection.closed
    assert not connection.committed
