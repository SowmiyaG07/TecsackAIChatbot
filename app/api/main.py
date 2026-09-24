from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.chat_service import answer_question, get_memory
from app.memory.conversation_history import MAX_SESSION_ID_LENGTH
from app.observability import configure_logging, create_trace_id


# Observability events go to the root logger; uvicorn only configures
# its own loggers, so without this they would not be printed.
configure_logging()

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Tecsack AI Chatbot",
    description="RAG-based company chatbot API",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SESSION_ID_LENGTH,
    )
    question: str = Field(..., min_length=1)

    @field_validator("session_id", "question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("must not be blank")

        return value


class ChatResponse(BaseModel):
    session_id: str
    trace_id: str
    answer: str


@app.get("/health")
def health() -> dict[str, Any]:
    """
    API health check. Redis is optional: the chatbot works without
    it (no conversation memory), so it does not make the API unhealthy.
    """

    memory = get_memory()

    if memory is None:
        redis_status = "not_configured"
    elif memory.ping():
        redis_status = "up"
    else:
        redis_status = "down"

    return {
        "status": "healthy",
        "redis": redis_status,
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Process one chatbot request.

    answer_question() runs the whole pipeline: Redis history, retrieval,
    context, generation, saving the turn to Redis, and logging the
    request to SQL Server - all under one trace_id. Redis problems are
    handled there and never fail the request.
    """

    trace_id = create_trace_id()

    try:
        result = answer_question(
            request.question,
            session_id=request.session_id,
            trace_id=trace_id,
        )

    except Exception:
        # The error is already logged and stored under this trace_id;
        # do not expose internal details (hosts, SQL errors) to clients.
        logger.exception("Chat request failed (trace_id=%s)", trace_id)

        raise HTTPException(
            status_code=500,
            detail={
                "message": "The chatbot could not answer this question.",
                "trace_id": trace_id,
            },
        ) from None

    return ChatResponse(
        session_id=request.session_id,
        trace_id=result.trace_id,
        answer=result.answer,
    )
