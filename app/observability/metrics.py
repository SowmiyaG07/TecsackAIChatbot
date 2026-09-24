from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

from app.observability.monitoring import format_error, record_event
from app.observability.trace import (
    LLMCallRecord,
    get_current_trace,
    get_current_trace_id,
)


# ---------------------------------------------------------
# Stage timing
# ---------------------------------------------------------

@contextmanager
def timed(
    stage: str,
    emit_event: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    Measure a pipeline stage with time.perf_counter().

    Durations accumulate on the current trace, so a stage that runs
    once per expanded query (embedding, vector_search, bm25) reports
    its total. Without an active trace this is a cheap no-op.

    With emit_event=True a "<stage>.completed" / "<stage>.failed"
    log line is written; extra fields can be added via the yielded
    dict:

        with timed("retrieval", emit_event=True) as span:
            results = ...
            span["chunks"] = len(results)
    """

    trace = get_current_trace()
    span: dict[str, Any] = {}
    started = time.perf_counter()

    try:
        yield span

    except BaseException as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000

        if trace is not None:
            _add_timing(trace, stage, elapsed_ms)

            # Innermost stage exits first, so it wins.
            if trace.failed_stage is None:
                trace.failed_stage = stage

        if emit_event or trace is not None and trace.failed_stage == stage:
            record_event(
                f"{stage}.failed",
                trace.trace_id if trace else None,
                level=logging.ERROR,
                latency_ms=round(elapsed_ms),
                error_type=type(exc).__name__,
            )

        raise

    elapsed_ms = (time.perf_counter() - started) * 1000

    if trace is not None:
        _add_timing(trace, stage, elapsed_ms)

    if emit_event:
        record_event(
            f"{stage}.completed",
            trace.trace_id if trace else None,
            latency_ms=round(elapsed_ms),
            **span,
        )


def _add_timing(trace, stage: str, elapsed_ms: float) -> None:
    trace.timings_ms[stage] = trace.timings_ms.get(stage, 0.0) + elapsed_ms
    trace.stage_counts[stage] = trace.stage_counts.get(stage, 0) + 1


# ---------------------------------------------------------
# Retrieval details
# ---------------------------------------------------------

def record_retrieval(
    intent: str,
    justification: str,
    metadata_filter: dict[str, Any] | None,
    expanded_queries: list[str],
    candidate_count: int,
    final_results: list[dict[str, Any]],
) -> None:
    """
    Attach retrieval details to the current trace. Never raises.
    """

    trace = get_current_trace()

    if trace is None:
        return

    try:
        trace.query_intent = intent
        trace.intent_justification = justification
        trace.metadata_filter = metadata_filter
        trace.expanded_queries = list(expanded_queries)
        trace.retrieved_chunk_count = candidate_count
        trace.final_top_k_count = len(final_results)
        trace.top_chunks = [
            {
                "id": result.get("id"),
                "rrf_score": round(float(result.get("rrf_score", 0.0)), 6),
            }
            for result in final_results
        ]

    except Exception as exc:
        record_event(
            "observability.internal_error",
            trace.trace_id,
            level=logging.ERROR,
            error=format_error(exc),
        )


# ---------------------------------------------------------
# LLM usage
# ---------------------------------------------------------

def _field(response: Any, name: str) -> Any:
    if isinstance(response, dict):
        return response.get(name)

    return getattr(response, name, None)


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_token_usage(
    response: Any,
) -> tuple[int | None, int | None, int | None]:
    """
    Actual token usage reported by Ollama:

        prompt_eval_count -> input tokens
        eval_count        -> output tokens

    A value Ollama did not report stays None (never estimated).
    total is only computed when both parts are known.
    """

    input_tokens = _as_int(_field(response, "prompt_eval_count"))
    output_tokens = _as_int(_field(response, "eval_count"))

    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )

    return input_tokens, output_tokens, total_tokens


def estimate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
) -> tuple[float | None, str | None]:
    """
    Cost from configured per-million-token rates. Local Ollama has
    no price, so this stays None unless rates are set in .env.
    """

    input_rate = os.getenv("LLM_INPUT_COST_PER_1M_TOKENS")
    output_rate = os.getenv("LLM_OUTPUT_COST_PER_1M_TOKENS")

    if (
        not input_rate
        or not output_rate
        or input_tokens is None
        or output_tokens is None
    ):
        return None, None

    cost = (
        input_tokens * float(input_rate)
        + output_tokens * float(output_rate)
    ) / 1_000_000

    return round(cost, 8), os.getenv("LLM_COST_CURRENCY", "USD")[:3]


def record_llm_usage(
    call_purpose: str,
    model_name: str,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    latency_ms: int,
    status: str = "success",
) -> None:
    """
    Log one LLM call and attach it to the current trace.
    Never raises.
    """

    trace = get_current_trace()

    try:
        estimated_cost, currency = estimate_cost(
            input_tokens,
            output_tokens,
        )

        if trace is not None:
            trace.llm_calls.append(
                LLMCallRecord(
                    call_purpose=call_purpose,
                    model_name=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    status=status,
                    estimated_cost=estimated_cost,
                    currency=currency,
                )
            )

            if call_purpose == "answer_generation":
                trace.model_name = model_name

        record_event(
            "llm.completed" if status == "success" else "llm.failed",
            get_current_trace_id(),
            level=logging.INFO if status == "success" else logging.ERROR,
            call_purpose=call_purpose,
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_usage=(
                "reported"
                if input_tokens is not None or output_tokens is not None
                else "unavailable"
            ),
            latency_ms=latency_ms,
        )

    except Exception as exc:
        record_event(
            "observability.internal_error",
            get_current_trace_id(),
            level=logging.ERROR,
            error=format_error(exc),
        )


def observed_chat(
    client: Any,
    call_purpose: str,
    **chat_kwargs: Any,
) -> Any:
    """
    Drop-in replacement for client.chat(**chat_kwargs) that records
    latency and token usage for the current trace.

    The response (or exception) is passed through unchanged.
    """

    model_name = str(chat_kwargs.get("model") or "unknown")
    started = time.perf_counter()

    try:
        response = client.chat(**chat_kwargs)

    except Exception:
        trace = get_current_trace()

        if trace is not None and trace.failed_stage is None:
            trace.failed_stage = call_purpose

        record_llm_usage(
            call_purpose=call_purpose,
            model_name=model_name,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            latency_ms=round((time.perf_counter() - started) * 1000),
            status="error",
        )

        raise

    latency_ms = round((time.perf_counter() - started) * 1000)

    try:
        input_tokens, output_tokens, total_tokens = extract_token_usage(
            response
        )
        model_name = str(_field(response, "model") or model_name)

    except Exception:
        input_tokens = output_tokens = total_tokens = None

    record_llm_usage(
        call_purpose=call_purpose,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
    )

    return response
