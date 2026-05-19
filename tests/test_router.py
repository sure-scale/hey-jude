import pytest
from hey_jude.models import ChatMessage
from hey_jude.services.router import _build_messages_with_preamble


def test_preamble_injected_as_system_message():
    messages = [
        ChatMessage(role="user", content="Tell me about Pinnacle Systems"),
    ]
    context_descriptors = {
        "Pinnacle Systems": "major technology corporation",
        "Vertex Holdings": "social media conglomerate",
    }
    result = _build_messages_with_preamble(messages, context_descriptors)
    assert result[0]["role"] == "system"
    assert "Pinnacle Systems: major technology corporation" in result[0]["content"]
    assert "Vertex Holdings: social media conglomerate" in result[0]["content"]
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "Tell me about Pinnacle Systems"


def test_preamble_prepended_before_existing_system():
    messages = [
        ChatMessage(role="system", content="You are a legal assistant."),
        ChatMessage(role="user", content="Tell me about Pinnacle Systems"),
    ]
    context_descriptors = {"Pinnacle Systems": "major technology corporation"}
    result = _build_messages_with_preamble(messages, context_descriptors)
    assert result[0]["role"] == "system"
    assert "Pinnacle Systems" in result[0]["content"]
    assert result[1]["role"] == "system"
    assert result[1]["content"] == "You are a legal assistant."


def test_empty_descriptors_no_preamble():
    messages = [
        ChatMessage(role="user", content="Hello"),
    ]
    result = _build_messages_with_preamble(messages, {})
    assert len(result) == 1
    assert result[0]["role"] == "user"
