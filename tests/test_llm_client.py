import json
import pytest
from hey_jude.config import Settings
from hey_jude.services.llm_client import (
    call_local_llm,
    parse_llm_response,
    ollama_native_chat_url,
)


def test_ollama_native_chat_url_standard():
    assert ollama_native_chat_url("http://localhost:11434/v1") == "http://localhost:11434/api/chat"


def test_ollama_native_chat_url_bare():
    assert ollama_native_chat_url("http://localhost:11434") == "http://localhost:11434/api/chat"


def test_ollama_native_chat_url_non_ollama():
    assert ollama_native_chat_url("http://localhost:8080/v1") is None


def test_parse_llm_response_valid_json():
    raw = json.dumps({"key": "value"})
    assert parse_llm_response(raw) == {"key": "value"}


def test_parse_llm_response_json_in_markdown():
    raw = '```json\n{"key": "value"}\n```'
    assert parse_llm_response(raw) == {"key": "value"}


def test_parse_llm_response_invalid():
    with pytest.raises(ValueError, match="Failed to parse"):
        parse_llm_response("not json")


async def test_call_local_llm_ollama_native(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={"message": {"role": "assistant", "content": '{"result": "ok"}'}},
    )
    result = await call_local_llm(
        "test prompt",
        Settings(local_llm_url="http://localhost:11434/v1"),
    )
    assert result == '{"result": "ok"}'
    request = httpx_mock.get_request()
    payload = json.loads(request.content)
    assert payload["stream"] is False
    assert payload["think"] is False


async def test_call_local_llm_openai_compatible(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:8080/v1/chat/completions",
        json={"choices": [{"message": {"content": '{"result": "ok"}'}}]},
    )
    result = await call_local_llm(
        "test prompt",
        Settings(local_llm_url="http://localhost:8080/v1"),
    )
    assert result == '{"result": "ok"}'
