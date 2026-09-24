"""
Redis conversation memory tests.

Uses the real Redis from REDIS_URL (WSL2). Each test run uses unique
session ids and deletes them afterwards. The whole module is skipped
when Redis is not reachable.

Ollama and ChromaDB are replaced with fakes.

    venv\\Scripts\\python -m pytest tests/test_redis_memory.py -v
"""

from __future__ import annotations

import json
import logging
import uuid

import pytest
import redis

import app.chat_service as chat_service
import app.generation.answer_generator as answer_generator
from app.generation.answer_generator import format_history
from app.memory import (
    RedisConversationMemory,
    create_redis_client,
    select_recent_messages,
)
from app.memory.token_utils import estimate_message_tokens


# ---------------------------------------------------------
# Redis availability
# ---------------------------------------------------------

def _redis_available() -> bool:
    try:
        return bool(create_redis_client().ping())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis (REDIS_URL) is not reachable",
)

# Closed port: refused on Linux, a ~REDIS_SOCKET_TIMEOUT_SECONDS
# connect timeout on Windows.
UNREACHABLE_REDIS_URL = "redis://127.0.0.1:1/0"


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def session_ids():
    """Unique session ids for this test; deleted afterwards."""

    prefix = f"pytest-{uuid.uuid4().hex[:12]}"
    created: list[str] = []

    def make(name: str) -> str:
        session_id = f"{prefix}-{name}"
        created.append(session_id)
        return session_id

    yield make

    client = create_redis_client()
    for session_id in created:
        client.delete(RedisConversationMemory.key(session_id))


@pytest.fixture
def memory():
    return RedisConversationMemory(max_messages=7, max_tokens=1500)


@pytest.fixture
def logs(caplog, monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("OBSERVABILITY_SQL_ENABLED", "false")
    caplog.set_level(logging.INFO, logger="tecsack.observability")
    return caplog


class CapturingLLM:
    """Fake Ollama client that records the answer-generation prompt."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.answer_prompts: list[str] = []

    def chat(self, **kwargs):
        messages = kwargs["messages"]
        self.answer_prompts.append(messages[-1]["content"])
        return {
            "model": kwargs["model"],
            "message": {"role": "assistant", "content": self.answers.pop(0)},
            "prompt_eval_count": 100,
            "eval_count": 10,
        }


def leadership_retriever(question, top_k):
    return [
        {
            "id": "doc_company_about_us_google_docs_chunk_00014",
            "text": (
                "Chief Executive Officer (CEO) - Sakthivel. "
                "Co-Founder & Chief Technology Officer (CTO) - Sowmiya."
            ),
            "metadata": {
                "file_name": "About US - Google Docs.pdf",
                "page_start": 6,
                "heading_path": "ABOUT US > 4. Leadership Team",
            },
            "rrf_score": 0.13,
        }
    ][:top_k]


def history_block(prompt: str) -> str:
    """The RECENT CONVERSATION section of a prompt ('' if absent)."""

    if "RECENT CONVERSATION" not in prompt:
        return ""

    return prompt.split("RECENT CONVERSATION", 1)[1].split(
        "DOCUMENT CONTEXT", 1
    )[0]


# ---------------------------------------------------------
# 1. Connection
# ---------------------------------------------------------

def test_ping(memory):
    assert memory.ping() is True


# ---------------------------------------------------------
# 2. Save and retrieve
# ---------------------------------------------------------

def test_save_and_retrieve_messages(memory, session_ids):
    session = session_ids("save")

    memory.add_message(session, "user", "Who is the CEO of Tecsack?")
    count = memory.add_message(
        session, "assistant", "Sakthivel is the CEO of Tecsack."
    )

    assert count == 2
    assert memory.get_message_count(session) == 2
    assert memory.get_messages(session) == [
        {"role": "user", "content": "Who is the CEO of Tecsack?"},
        {"role": "assistant", "content": "Sakthivel is the CEO of Tecsack."},
    ]


def test_stored_as_json_under_session_key(memory, session_ids):
    session = session_ids("json")
    memory.add_message(session, "user", "Hello")

    raw = create_redis_client().lrange(f"chat:history:{session}", 0, -1)

    assert [json.loads(item) for item in raw] == [
        {"role": "user", "content": "Hello"}
    ]


def test_session_key_expires(memory, session_ids):
    session = session_ids("ttl")
    memory.add_message(session, "user", "Hello")

    ttl = create_redis_client().ttl(memory.key(session))

    assert 0 < ttl <= memory.ttl_seconds


def test_invalid_input_rejected(memory, session_ids):
    with pytest.raises(ValueError):
        memory.add_message(session_ids("bad-role"), "system", "x")

    with pytest.raises(ValueError):
        memory.add_message("", "user", "x")

    with pytest.raises(ValueError):
        memory.add_message(session_ids("empty"), "user", "   ")


# ---------------------------------------------------------
# 3. Session isolation
# ---------------------------------------------------------

def test_sessions_are_isolated(memory, session_ids):
    session_a = session_ids("A")
    session_b = session_ids("B")

    memory.add_message(session_a, "user", "question from A")
    memory.add_message(session_b, "user", "question from B")

    assert memory.get_messages(session_a) == [
        {"role": "user", "content": "question from A"}
    ]
    assert memory.get_messages(session_b) == [
        {"role": "user", "content": "question from B"}
    ]


# ---------------------------------------------------------
# 4. Maximum message count
# ---------------------------------------------------------

def test_only_newest_max_messages_kept(memory, session_ids):
    session = session_ids("max")

    for index in range(1, 11):
        role = "user" if index % 2 else "assistant"
        memory.add_message(session, role, f"message {index}")

    assert memory.get_message_count(session) == 7
    assert [m["content"] for m in memory.get_messages(session)] == [
        f"message {index}" for index in range(4, 11)
    ]


@pytest.mark.parametrize(
    "limits",
    [{"max_messages": 0}, {"max_messages": -3}, {"max_tokens": 0}],
)
def test_non_positive_limits_rejected(limits):
    # LTRIM key -0 -1 would keep everything, so 0 must not be accepted.
    with pytest.raises(ValueError):
        RedisConversationMemory(**limits)


# ---------------------------------------------------------
# 5. Token limit
# ---------------------------------------------------------

def test_history_respects_default_token_budget(memory, session_ids):
    session = session_ids("tokens")

    # Seven ~400-token messages (~2800 tokens) against a 1500 budget.
    for index in range(7):
        memory.add_message(
            session,
            "user" if index % 2 == 0 else "assistant",
            f"message-{index} " + "word " * 400,
        )

    selected = memory.get_messages_for_llm(session)
    used = sum(estimate_message_tokens(m) for m in selected)

    assert used <= 1500
    assert 0 < len(selected) < 7
    # The newest messages are the ones kept.
    assert selected[-1]["content"].startswith("message-6")


def test_selection_stops_at_first_message_over_budget():
    messages = [
        {"role": "user", "content": "old " * 50},
        {"role": "assistant", "content": "huge " * 500},
        {"role": "user", "content": "recent question"},
    ]

    selected = select_recent_messages(messages, max_messages=7, max_tokens=100)

    # The huge message does not fit, so nothing older is taken either.
    assert selected == [messages[2]]


def test_newest_message_larger_than_budget_gives_empty_history():
    messages = [{"role": "user", "content": "word " * 500}]

    assert select_recent_messages(messages, 7, 100) == []


# ---------------------------------------------------------
# 6. Chronological order
# ---------------------------------------------------------

def test_llm_history_is_chronological(memory, session_ids):
    session = session_ids("order")

    for index in range(1, 6):
        memory.add_message(
            session,
            "user" if index % 2 else "assistant",
            f"message {index}",
        )

    assert [m["content"] for m in memory.get_messages_for_llm(session)] == [
        "message 1",
        "message 2",
        "message 3",
        "message 4",
        "message 5",
    ]


# ---------------------------------------------------------
# 7. Empty session
# ---------------------------------------------------------

def test_empty_session_returns_empty_list(memory, session_ids):
    session = session_ids("empty-session")

    assert memory.get_messages(session) == []
    assert memory.get_messages_for_llm(session) == []
    assert memory.get_message_count(session) == 0


# ---------------------------------------------------------
# 8. Redis unavailable
# ---------------------------------------------------------

def test_ping_false_when_redis_unavailable():
    memory = RedisConversationMemory(
        client=create_redis_client(UNREACHABLE_REDIS_URL)
    )

    assert memory.ping() is False

    # Refused (ConnectionError) or, on Windows, a connect timeout
    # (TimeoutError) - the chatbot handles both.
    with pytest.raises((redis.ConnectionError, redis.TimeoutError)):
        memory.get_messages("any-session")


def test_chatbot_answers_when_redis_unavailable(monkeypatch, logs):
    llm = CapturingLLM(["The CEO is Sakthivel."])
    monkeypatch.setattr(answer_generator, "ollama_client", llm)

    unavailable = RedisConversationMemory(
        client=create_redis_client(UNREACHABLE_REDIS_URL)
    )

    result = chat_service.answer_question(
        "Who is the CEO of Tecsack?",
        session_id="offline-session",
        top_k=5,
        retriever=leadership_retriever,
        memory=unavailable,
    )

    assert result.answer == "The CEO is Sakthivel."
    assert result.history == []
    assert "RECENT CONVERSATION" not in llm.answer_prompts[0]

    # One failure logged for the load; the save is skipped.
    failures = [
        r.getMessage() for r in logs.records
        if "event=redis.connection.failed" in r.getMessage()
    ]
    assert len(failures) == 1
    assert "operation=history_load" in failures[0]
    assert "event=request.completed" in logs.text


def test_answer_returned_when_saving_history_fails(
    memory, session_ids, monkeypatch, logs,
):
    session = session_ids("save-fails")
    llm = CapturingLLM(["The CEO is Sakthivel."])
    monkeypatch.setattr(answer_generator, "ollama_client", llm)

    def broken_save(session_id, messages):
        raise redis.ConnectionError("Redis went away")

    monkeypatch.setattr(memory, "add_messages", broken_save)

    result = chat_service.answer_question(
        "Who is the CEO of Tecsack?",
        session_id=session,
        top_k=5,
        retriever=leadership_retriever,
        memory=memory,
    )

    assert result.answer == "The CEO is Sakthivel."
    assert "operation=message_save" in logs.text
    assert "event=request.completed" in logs.text


# ---------------------------------------------------------
# 9. Clear history
# ---------------------------------------------------------

def test_clear_history_only_affects_that_session(memory, session_ids, logs):
    session_a = session_ids("clear-A")
    session_b = session_ids("clear-B")

    memory.add_message(session_a, "user", "A")
    memory.add_message(session_b, "user", "B")

    assert chat_service.clear_session_history(session_a, memory) is True

    assert memory.get_messages(session_a) == []
    assert memory.get_messages(session_b) == [{"role": "user", "content": "B"}]
    assert "event=redis.history.cleared" in logs.text


# ---------------------------------------------------------
# History reaches the LLM (pipeline integration)
# ---------------------------------------------------------

def test_follow_up_question_receives_previous_turn(
    memory, session_ids, monkeypatch, logs,
):
    session = session_ids("conversation")
    llm = CapturingLLM([
        "Sakthivel is the CEO of Tecsack.",
        "Sowmiya is the CTO.",
    ])
    monkeypatch.setattr(answer_generator, "ollama_client", llm)

    def ask(question):
        return chat_service.answer_question(
            question,
            session_id=session,
            top_k=5,
            retriever=leadership_retriever,
            memory=memory,
        )

    first = ask("Who is the CEO of Tecsack?")
    second = ask("What about the CTO?")

    # First request: no history block at all.
    assert first.history == []
    assert "RECENT CONVERSATION" not in llm.answer_prompts[0]

    # Second request: previous turn, oldest first, and the current
    # question only once (as USER QUESTION, not in the history).
    assert second.history == [
        {"role": "user", "content": "Who is the CEO of Tecsack?"},
        {"role": "assistant", "content": "Sakthivel is the CEO of Tecsack."},
    ]
    block = history_block(llm.answer_prompts[1])
    assert "User: Who is the CEO of Tecsack?" in block
    assert "Assistant: Sakthivel is the CEO of Tecsack." in block
    assert "What about the CTO?" not in block
    assert llm.answer_prompts[1].count("What about the CTO?") == 1

    # Documents still come after the history, and both turns are stored.
    prompt = llm.answer_prompts[1]
    assert prompt.index("RECENT CONVERSATION") < prompt.index("DOCUMENT CONTEXT")
    assert memory.get_message_count(session) == 4

    assert "event=redis.history.loaded" in logs.text
    assert "message_count=2" in logs.text
    assert "event=redis.message.saved" in logs.text


def test_logs_do_not_contain_message_content(
    memory, session_ids, monkeypatch, logs,
):
    session = session_ids("private")
    llm = CapturingLLM(["Private answer text"])
    monkeypatch.setattr(answer_generator, "ollama_client", llm)

    chat_service.answer_question(
        "Private question text",
        session_id=session,
        top_k=5,
        retriever=leadership_retriever,
        memory=memory,
    )

    assert "Private question text" not in logs.text
    assert "Private answer text" not in logs.text


def test_prompt_unchanged_without_history():
    assert format_history(None) == ""
    assert format_history([]) == ""
