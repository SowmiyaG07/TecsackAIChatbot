"""
Request tracing, LLM usage and latency metrics for the RAG pipeline.

Typical use (see app/chat_service.py):

    with start_trace(question, session_id) as trace:
        with timed("retrieval", emit_event=True):
            ...
        response = observed_chat(client, "answer_generation", model=..., messages=...)
"""

from app.observability.metrics import (
    estimate_cost,
    extract_token_usage,
    observed_chat,
    record_llm_usage,
    record_retrieval,
    timed,
)
from app.observability.monitoring import (
    SqlServerObservabilityStore,
    configure_logging,
    finish_trace,
    format_error,
    get_store,
    is_observability_enabled,
    persist_trace,
    record_event,
    set_store,
    start_trace,
)
from app.observability.trace import (
    LLMCallRecord,
    RequestTrace,
    create_trace_id,
    get_current_trace,
    get_current_trace_id,
)

__all__ = [
    "LLMCallRecord",
    "RequestTrace",
    "SqlServerObservabilityStore",
    "configure_logging",
    "create_trace_id",
    "estimate_cost",
    "extract_token_usage",
    "finish_trace",
    "format_error",
    "get_current_trace",
    "get_current_trace_id",
    "get_store",
    "is_observability_enabled",
    "observed_chat",
    "persist_trace",
    "record_event",
    "record_llm_usage",
    "record_retrieval",
    "set_store",
    "start_trace",
    "timed",
]
