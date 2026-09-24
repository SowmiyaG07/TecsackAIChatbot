from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------
# Time
# ---------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------
# Records
# ---------------------------------------------------------

@dataclass
class LLMCallRecord:
    """
    One LLM call (query_expansion, answer_generation, ...).

    Token counts are None when the model server did not report
    them. They are never estimated.
    """

    call_purpose: str
    model_name: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: int
    status: str = "success"
    estimated_cost: float | None = None
    currency: str | None = None
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class RequestTrace:
    """
    Everything observed for one chatbot request.

    One request == one trace_id. Every LLM call, timing and the
    final ConversationHistory row share this trace_id.
    """

    trace_id: str
    question: str
    session_id: str | None = None

    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None

    answer: str | None = None
    model_name: str | None = None

    status: str = "in_progress"
    result_type: str | None = None
    error_message: str | None = None

    # First (innermost) stage that raised, e.g. "query_expansion".
    failed_stage: str | None = None

    # Retrieval
    query_intent: str | None = None
    intent_justification: str | None = None
    metadata_filter: dict[str, Any] | None = None
    expanded_queries: list[str] = field(default_factory=list)
    retrieved_chunk_count: int | None = None
    final_top_k_count: int | None = None
    top_chunks: list[dict[str, Any]] = field(default_factory=list)

    # Conversation memory (Redis)
    # loaded | empty | no_session | unavailable
    memory_status: str | None = None
    history_message_count: int | None = None
    history_token_estimate: int | None = None

    # Accumulated per stage; a stage may run several times
    # (e.g. one embedding per expanded query).
    timings_ms: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)

    llm_calls: list[LLMCallRecord] = field(default_factory=list)

    # Monotonic start for latency; wall-clock times are only
    # used for created_at / completed_at.
    start_perf: float = field(default_factory=time.perf_counter)

    @property
    def total_latency_ms(self) -> int | None:
        if self.completed_at is None:
            return None

        return self.timing("request")

    def timing(self, *stages: str) -> int | None:
        """
        Sum of the given stages in whole milliseconds, or None
        when none of them ran.
        """

        values = [
            self.timings_ms[stage]
            for stage in stages
            if stage in self.timings_ms
        ]

        if not values:
            return None

        return round(sum(values))


# ---------------------------------------------------------
# Current trace (per request / per async task)
# ---------------------------------------------------------

_current_trace: ContextVar[RequestTrace | None] = ContextVar(
    "tecsack_current_trace",
    default=None,
)


def create_trace_id() -> str:
    return str(uuid.uuid4())


def get_current_trace() -> RequestTrace | None:
    return _current_trace.get()


def get_current_trace_id() -> str | None:
    trace = _current_trace.get()

    return trace.trace_id if trace else None


def _set_current_trace(trace: RequestTrace | None):
    return _current_trace.set(trace)


def _reset_current_trace(token) -> None:
    _current_trace.reset(token)
