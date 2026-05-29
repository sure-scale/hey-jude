import json
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import ASGITransport, AsyncClient

from hey_jude.main import create_app
from hey_jude.config import Settings
from hey_jude.models import (
    AnonymizationResult,
    ChatMessage,
    DetectedEntity,
    FoundEntity,
    SubstitutionResult,
)


@pytest.fixture
def test_settings():
    return Settings(
        redis_url="redis://localhost:6379/1",
        redis_ttl_seconds=60,
        api_key="sk-test-key",
        anonymization_mode="mechanical",
    )


@pytest.fixture
async def app(test_settings):
    application = create_app(test_settings)
    yield application


@pytest.fixture
async def client(app):
    async with app.router.lifespan_context(app):
        app.state.redis_client = MemoryRedis()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.fixture(autouse=True)
def mock_litellm():
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a mocked response.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "total_tokens": 10,
            },
        }
        mock.return_value = mock_resp
        yield mock


@pytest.fixture(autouse=True)
def mock_local_llm():
    with patch("hey_jude.services.substitutor._call_local_llm", new_callable=AsyncMock) as mock:
        mock.return_value = json.dumps(
            {
                "sensitivity": "low",
                "reasoning": "Mocked local anonymization pass.",
                "mapping": {},
                "context_descriptors": {},
                "sanitized_text": "hello",
                "needs_clarification": False,
                "clarification_question": None,
            }
        )
        yield mock


async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "redis" in data
    assert "local_llm" in data


async def test_auth_missing_key(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 401


async def test_auth_wrong_key(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


async def test_auth_valid_key_bearer(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "The weather is nice."}],
        },
        headers={"Authorization": "Bearer sk-test-key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "chatcmpl-mock"
    assert data["choices"][0]["message"]["content"] == "This is a mocked response."


async def test_entity_free_requests_still_run_local_llm_pass(client, app):
    local_llm_response = {
        "sensitivity": "low",
        "reasoning": "No sensitive entities detected, but the request was still inspected.",
        "mapping": {},
        "context_descriptors": {},
        "sanitized_text": "hello",
        "needs_clarification": False,
        "clarification_question": None,
    }

    app.state.redis_client = MemoryRedis()

    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local_llm, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = []
        mock_local_llm.return_value = json.dumps(local_llm_response)
        mock_route.return_value = {
            "id": "chatcmpl-no-entities",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
        }

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    assert mock_local_llm.await_count == 1
    assert mock_route.await_args.kwargs["messages"][0].content == "hello"
    assert resp.json()["heyjude_metadata"]["entities_detected"] == 0


async def test_malformed_request(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={"not": "valid"},
        headers={"X-API-Key": "sk-test-key"},
    )
    assert resp.status_code == 422


async def test_completion_params_forwarded_to_router(client):
    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = []
        mock_route.return_value = {
            "id": "chatcmpl-params",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.2,
                "max_tokens": 50,
                "top_p": 0.8,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.3,
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    assert mock_route.await_args.kwargs["temperature"] == 0.2
    assert mock_route.await_args.kwargs["max_tokens"] == 50
    assert mock_route.await_args.kwargs["top_p"] == 0.8
    assert mock_route.await_args.kwargs["frequency_penalty"] == 0.1
    assert mock_route.await_args.kwargs["presence_penalty"] == 0.3


async def test_structured_openai_envelopes_forwarded_unchanged(client):
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "case_summary",
            "schema": {
                "type": "object",
                "properties": {
                    "risk": {"type": "string"},
                    "next_steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["risk", "next_steps"],
            },
        },
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_document",
                "description": "Find a document by ID",
                "parameters": {
                    "type": "object",
                    "properties": {"document_id": {"type": "string"}},
                    "required": ["document_id"],
                },
            },
        }
    ]

    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = []
        mock_route.return_value = {
            "id": "chatcmpl-structured",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "{\"risk\":\"low\",\"next_steps\":[]}"},
                    "finish_reason": "stop",
                }
            ],
        }

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Summarize Microsoft."}],
                "response_format": response_format,
                "tools": tools,
                "tool_choice": "auto",
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    assert mock_route.await_args.kwargs["response_format"] == response_format
    assert mock_route.await_args.kwargs["tools"] == tools
    assert mock_route.await_args.kwargs["tool_choice"] == "auto"


async def test_chat_endpoint_extracts_html_file_content(client):
    encoded = "PGh0bWw+PGJvZHk+PHA+Sm9obiBTbWl0aCBhdCBNYWlubGFuZCBDb3JwLjwvcD48L2JvZHk+PC9odG1sPg=="

    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = []
        mock_route.return_value = {
            "id": "chatcmpl-html",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Summarize the attachment."},
                            {
                                "type": "input_file",
                                "filename": "memo.html",
                                "file_data": f"data:text/html;base64,{encoded}",
                            },
                        ],
                    }
                ],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    routed_messages = mock_route.await_args.kwargs["messages"]
    assert "Summarize the attachment." in routed_messages[0].content
    assert "John Smith at Mainland Corp." in routed_messages[0].content


async def test_chat_endpoint_rejects_unreadable_image_by_default(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,bm90LXJlYWQ=",
                            },
                        }
                    ],
                }
            ],
        },
        headers={"X-API-Key": "sk-test-key"},
    )

    assert resp.status_code == 422
    assert "not readable as text" in resp.json()["detail"]


async def test_chat_endpoint_warns_and_omits_unreadable_image(client, app):
    app.state.settings.document_unreadable_action = "warn"

    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = []
        mock_route.return_value = {
            "id": "chatcmpl-image-warn",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Review this."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,bm90LXJlYWQ=",
                                },
                            },
                        ],
                    }
                ],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["heyjude_metadata"]["document_warnings"][0]["action"] == "warn"
    routed = mock_route.await_args.kwargs["messages"][0].content
    assert "Review this." in routed
    assert "not-read" not in routed


async def test_responses_endpoint_accepts_mike_openai_shape(client):
    responses_tools = [
        {
            "type": "function",
            "name": "lookup_document",
            "description": "Find a document by ID",
            "parameters": {
                "type": "object",
                "properties": {"document_id": {"type": "string"}},
                "required": ["document_id"],
            },
        }
    ]

    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = []
        mock_route.return_value = {
            "id": "chatcmpl-responses",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Responses response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "total_tokens": 12,
            },
        }

        resp = await client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.4-nano",
                "instructions": "You are concise.",
                "input": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "previous answer"},
                    {
                        "type": "function_call_output",
                        "call_id": "call_123",
                        "output": "tool result",
                    },
                ],
                "tools": responses_tools,
                "tool_choice": "auto",
                "max_output_tokens": 64,
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "response"
    assert data["model"] == "gpt-5.4-nano"
    assert data["output_text"] == "Responses response"
    assert data["output"][0]["content"][0]["text"] == "Responses response"
    assert data["usage"] == {
        "input_tokens": 8,
        "output_tokens": 4,
        "total_tokens": 12,
    }
    routed_messages = mock_route.await_args.kwargs["messages"]
    assert [message.role for message in routed_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [message.content for message in routed_messages] == [
        "You are concise.",
        "hello",
        "previous answer",
        "tool result",
    ]
    assert mock_route.await_args.kwargs["max_tokens"] == 64
    assert mock_route.await_args.kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_document",
                "description": "Find a document by ID",
                "parameters": responses_tools[0]["parameters"],
            },
        }
    ]
    assert mock_route.await_args.kwargs["tool_choice"] == "auto"


async def test_tool_call_arguments_are_deanonymized(client, app):
    app.state.redis_client = MemoryRedis()
    sub_result = SubstitutionResult(
        mapping={"John Smith": "Michael Jones"},
        reverse_mapping={"Michael Jones": "John Smith"},
        context_descriptors={},
        sanitized_messages=[
            ChatMessage(role="user", content="Find cases for Michael Jones")
        ],
        sensitivity="low",
        needs_clarification=False,
        clarification_question=None,
    )

    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.routes.substitute_entities", new_callable=AsyncMock
    ) as mock_substitute, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = [
            DetectedEntity(
                text="John Smith",
                entity_type="PERSON",
                start=15,
                end=25,
                score=0.95,
            )
        ]
        mock_substitute.return_value = sub_result
        mock_route.return_value = {
            "id": "chatcmpl-toolcall",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Looking up Michael Jones.",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "search_cases",
                                    "arguments": json.dumps(
                                        {"client_name": "Michael Jones", "year": 2024}
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Find cases for John Smith"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_cases",
                            "description": "Search case database",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "client_name": {"type": "string"},
                                    "year": {"type": "integer"},
                                },
                            },
                        },
                    }
                ],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Looking up John Smith."
    tool_call = data["choices"][0]["message"]["tool_calls"][0]
    args = json.loads(tool_call["function"]["arguments"])
    assert args["client_name"] == "John Smith"
    assert args["year"] == 2024


async def test_responses_stream_endpoint_emits_openai_sse(client):
    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = []
        mock_route.return_value = {
            "id": "chatcmpl-responses-stream",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "stream text"},
                    "finish_reason": "stop",
                }
            ],
        }

        resp = await client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.4-nano",
                "input": "hello",
                "stream": True,
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert '"type": "response.created"' in body
    assert '"type": "response.output_text.delta"' in body
    assert '"delta": "stream text"' in body
    assert '"type": "response.completed"' in body
    assert "data: [DONE]" in body


async def test_anthropic_messages_endpoint_accepts_native_shape(client):
    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = []
        mock_route.return_value = {
            "id": "chatcmpl-anthropic",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Anthropic response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        }

        resp = await client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-latest",
                "max_tokens": 128,
                "system": "You are a legal assistant.",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.2,
                "top_p": 0.8,
            },
            headers={"x-api-key": "sk-test-key", "anthropic-version": "2023-06-01"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["model"] == "claude-3-5-sonnet-latest"
    assert data["content"] == [{"type": "text", "text": "Anthropic response"}]
    assert data["usage"] == {"input_tokens": 7, "output_tokens": 3}
    routed_messages = mock_route.await_args.kwargs["messages"]
    assert [message.role for message in routed_messages] == ["system", "user"]
    assert [message.content for message in routed_messages] == [
        "You are a legal assistant.",
        "hello",
    ]
    assert mock_route.await_args.kwargs["max_tokens"] == 128


async def test_gemini_generate_content_endpoint_accepts_native_shape(client):
    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = []
        mock_route.return_value = {
            "id": "chatcmpl-gemini",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Gemini response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "total_tokens": 12,
            },
        }

        resp = await client.post(
            "/v1beta/models/gemini-1.5-flash:generateContent",
            json={
                "systemInstruction": {"parts": [{"text": "You are concise."}]},
                "contents": [
                    {"role": "user", "parts": [{"text": "hello"}]},
                    {"role": "model", "parts": [{"text": "previous answer"}]},
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.7,
                    "maxOutputTokens": 64,
                },
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["candidates"][0]["content"]["role"] == "model"
    assert data["candidates"][0]["content"]["parts"] == [{"text": "Gemini response"}]
    assert data["usageMetadata"] == {
        "promptTokenCount": 8,
        "candidatesTokenCount": 4,
        "totalTokenCount": 12,
    }
    routed_messages = mock_route.await_args.kwargs["messages"]
    assert [message.role for message in routed_messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert [message.content for message in routed_messages] == [
        "You are concise.",
        "hello",
        "previous answer",
    ]
    assert mock_route.await_args.kwargs["max_tokens"] == 64


async def test_end_to_end_anonymization_pipeline(client):
    local_llm_response = {
        "sensitivity": "low",
        "reasoning": "Standard mapping of company and person.",
        "mapping": {
            "John Smith": "Richard Roe",
            "Microsoft": "Pinnacle Systems",
        },
        "context_descriptors": {
            "Richard Roe": "a legal professional",
            "Pinnacle Systems": "a major technology corporation",
        },
        "sanitized_text": "Richard Roe is a lawyer at Pinnacle Systems. Contact him at user_1@example.com.",
        "needs_clarification": False,
        "clarification_question": None,
    }

    mock_external_resp = {
        "id": "chatcmpl-mock-e2e",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I recommend contacting Richard Roe at Pinnacle Systems.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 10,
            "total_tokens": 25,
        },
    }

    with patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local_call, patch(
        "litellm.acompletion", new_callable=AsyncMock
    ) as mock_external_call:

        mock_local_call.return_value = json.dumps(local_llm_response)

        mock_resp_obj = MagicMock()
        mock_resp_obj.model_dump.return_value = mock_external_resp
        mock_external_call.return_value = mock_resp_obj

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": "John Smith is a lawyer at Microsoft. Contact him at john@microsoft.com.",
                    }
                ],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

        assert resp.status_code == 200
        data = resp.json()

        # Verify the returned content is fully reverse-mapped back to original entities!
        assert "John Smith" in data["choices"][0]["message"]["content"]
        assert "Microsoft" in data["choices"][0]["message"]["content"]
        assert "Richard Roe" not in data["choices"][0]["message"]["content"]
        assert (
            "Pinnacle Systems" not in data["choices"][0]["message"]["content"]
        )

        # Check heyjude metadata
        meta = data["heyjude_metadata"]
        assert meta["entities_detected"] >= 2
        assert meta["sensitivity"] == "low"
        assert meta["status"] == "completed"


class MemoryRedis:
    def __init__(self, store_mappings: bool = True):
        self.store_mappings = store_mappings
        self.mappings = {}
        self.requests = {}

    async def health_check(self):
        return True

    async def store_mapping(self, request_id, mapping, ttl):
        if self.store_mappings:
            self.mappings[request_id] = mapping

    async def get_mapping(self, request_id):
        return self.mappings.get(request_id)

    async def store_request(self, request_id, request_data, ttl):
        self.requests[request_id] = request_data

    async def get_request(self, request_id):
        return self.requests.get(request_id)

    async def close(self):
        return None


async def test_reverse_mapping_uses_redis_and_returns_504_when_mapping_missing(client, app):
    app.state.redis_client = MemoryRedis(store_mappings=False)
    sub_result = SubstitutionResult(
        mapping={"Microsoft": "Pinnacle Systems"},
        reverse_mapping={"Pinnacle Systems": "Microsoft"},
        context_descriptors={},
        sanitized_messages=[
            ChatMessage(role="user", content="Tell me about Pinnacle Systems")
        ],
        sensitivity="low",
        needs_clarification=False,
        clarification_question=None,
    )

    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.routes.substitute_entities", new_callable=AsyncMock
    ) as mock_substitute, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = [
            DetectedEntity(
                text="Microsoft",
                entity_type="ORGANIZATION",
                start=14,
                end=23,
                score=0.9,
            )
        ]
        mock_substitute.return_value = sub_result
        mock_route.return_value = {
            "id": "chatcmpl-missing-map",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Pinnacle Systems has relevant cases.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Tell me about Microsoft"}],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 504


async def test_anonymization_validation_failure_returns_502(client, app):
    app.state.redis_client = MemoryRedis()

    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.routes.substitute_entities", new_callable=AsyncMock
    ) as mock_substitute, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = [
            DetectedEntity(
                text="Microsoft",
                entity_type="ORGANIZATION",
                start=14,
                end=23,
                score=0.9,
            )
        ]
        mock_substitute.side_effect = ValueError(
            "Sanitized message structure mismatch"
        )

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Tell me about Microsoft"}],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 502
    assert resp.json()["detail"] == "Anonymization validation failed"
    mock_route.assert_not_awaited()


async def test_clarification_followup_retrieves_original_context(client, app):
    app.state.redis_client = MemoryRedis()
    clarify_result = SubstitutionResult(
        mapping={},
        reverse_mapping={},
        context_descriptors={},
        sanitized_messages=[],
        sensitivity="high",
        needs_clarification=True,
        clarification_question="Which dispute do you mean?",
    )
    completed_result = SubstitutionResult(
        mapping={"Microsoft": "Pinnacle Systems"},
        reverse_mapping={"Pinnacle Systems": "Microsoft"},
        context_descriptors={},
        sanitized_messages=[
            ChatMessage(role="user", content="Original sanitized"),
            ChatMessage(role="user", content="Clarification sanitized"),
        ],
        sensitivity="high",
        needs_clarification=False,
        clarification_question=None,
    )

    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.routes.substitute_entities", new_callable=AsyncMock
    ) as mock_substitute, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = [
            DetectedEntity(
                text="Microsoft",
                entity_type="ORGANIZATION",
                start=14,
                end=23,
                score=0.9,
            )
        ]
        mock_substitute.side_effect = [clarify_result, completed_result]
        mock_route.return_value = {
            "id": "chatcmpl-followup",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Pinnacle Systems answer."},
                    "finish_reason": "stop",
                }
            ],
        }

        first = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Tell me about Microsoft"}],
            },
            headers={"X-API-Key": "sk-test-key"},
        )
        request_id = first.json()["heyjude_metadata"]["request_id"]

        second = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "request_id": request_id,
                "messages": [{"role": "user", "content": "The licensing dispute."}],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    second_messages = mock_substitute.await_args_list[1].args[0]
    assert [m.content for m in second_messages] == [
        "Tell me about Microsoft",
        "The licensing dispute.",
    ]


async def test_llm_mode_anonymization_pipeline(client, app):
    app.state.redis_client = MemoryRedis()
    app.state.settings.anonymization_mode = "llm"
    app.state.settings.safety_net_strictness = "off"
    app.state.anonymization_prompt = Path("prompts/anonymize.txt").read_text()

    llm_response = json.dumps({
        "entities_found": [
            {"text": "Microsoft", "type": "ORGANIZATION", "action": "replace",
             "replacement": "SOFTWARE_COMPANY_01", "reason": "real company"},
            {"text": "Purchaser", "type": "DEFINED_TERM", "action": "keep",
             "replacement": None, "reason": "legal defined term"},
        ],
        "context_descriptors": {"SOFTWARE_COMPANY_01": "tech company"},
        "sensitivity": "low",
    })

    with patch(
        "hey_jude.services.anonymizer.call_local_llm",
        new_callable=AsyncMock,
    ) as mock_llm, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_llm.return_value = llm_response
        mock_route.return_value = {
            "id": "chatcmpl-llm-mode",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The SOFTWARE_COMPANY_01 breach claim by the Purchaser is valid.",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "The Purchaser sued Microsoft for breach"}],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "Microsoft" in data["choices"][0]["message"]["content"]
    assert "Purchaser" in data["choices"][0]["message"]["content"]
    assert "SOFTWARE_COMPANY_01" not in data["choices"][0]["message"]["content"]

    routed = mock_route.await_args.kwargs["messages"]
    assert "SOFTWARE_COMPANY_01" in routed[0].content
    assert "Purchaser" in routed[0].content


async def test_llm_mode_strict_safety_net_blocks(client, app):
    app.state.redis_client = MemoryRedis()
    app.state.settings.anonymization_mode = "llm"
    app.state.settings.safety_net_strictness = "strict"
    app.state.anonymization_prompt = Path("prompts/anonymize.txt").read_text()

    llm_response = json.dumps({
        "entities_found": [],
        "context_descriptors": {},
        "sensitivity": "low",
    })

    with patch(
        "hey_jude.services.anonymizer.call_local_llm",
        new_callable=AsyncMock,
    ) as mock_llm, patch(
        "hey_jude.services.safety_net.detect_entities",
        new_callable=AsyncMock,
    ) as mock_detect:
        mock_llm.return_value = llm_response
        mock_detect.return_value = [
            DetectedEntity(text="John Smith", entity_type="PERSON",
                          start=0, end=10, score=0.9),
        ]

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "John Smith filed a claim"}],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 422
    assert "Safety net" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_mechanical_mode_unchanged(client, app):
    """Regression: mechanical mode produces identical results to pre-refactor behavior."""
    app.state.redis_client = MemoryRedis()
    app.state.settings.anonymization_mode = "mechanical"

    local_llm_response = json.dumps({
        "sensitivity": "low",
        "reasoning": "Standard mapping.",
        "mapping": {
            "John Smith": "Richard Roe",
            "Microsoft": "Pinnacle Systems",
        },
        "context_descriptors": {
            "Richard Roe": "a legal professional",
            "Pinnacle Systems": "a technology corporation",
        },
        "sanitized_text": "",
        "needs_clarification": False,
        "clarification_question": None,
    })

    with patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_local.return_value = local_llm_response
        mock_route.return_value = {
            "id": "chatcmpl-mechanical",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Richard Roe at Pinnacle Systems.",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "John Smith works at Microsoft"}],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "John Smith" in data["choices"][0]["message"]["content"]
    assert "Microsoft" in data["choices"][0]["message"]["content"]


async def test_known_entity_guaranteed_replaced_mechanical(client, app):
    """A known entity is replaced even if detection and the local LLM miss it."""
    from hey_jude.services.known_entities import KnownEntity

    app.state.redis_client = MemoryRedis()
    app.state.settings.anonymization_mode = "mechanical"
    app.state.known_entities = [
        KnownEntity(entity_type="CLIENT_NAME", term="Acme Corp", replace_with="CLIENT_01")
    ]

    empty_local = json.dumps({
        "sensitivity": "low",
        "reasoning": "Nothing detected by the model.",
        "mapping": {},
        "context_descriptors": {},
        "sanitized_text": "",
        "needs_clarification": False,
        "clarification_question": None,
    })

    with patch("hey_jude.routes.detect_entities", new_callable=AsyncMock) as mock_detect, patch(
        "hey_jude.services.substitutor._call_local_llm", new_callable=AsyncMock
    ) as mock_local, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        mock_detect.return_value = []
        mock_local.return_value = empty_local
        mock_route.return_value = {
            "id": "chatcmpl-known",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "CLIENT_01 is the client."},
                "finish_reason": "stop",
            }],
        }

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Engagement with Acme Corp."}],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    # The prompt that left the gateway must not contain the real client name.
    sent = mock_route.await_args.kwargs["messages"][0].content
    assert "Acme Corp" not in sent
    assert "CLIENT_01" in sent
    # The response is de-anonymized back to the real name.
    assert resp.json()["choices"][0]["message"]["content"] == "Acme Corp is the client."


async def test_known_entity_seeds_llm_mode_mapping(client, app):
    """In LLM mode the known seed flows through and is guaranteed applied."""
    from hey_jude.services.known_entities import KnownEntity
    from hey_jude.models import SafetyNetResult

    app.state.redis_client = MemoryRedis()
    app.state.settings.anonymization_mode = "llm"
    app.state.known_entities = [
        KnownEntity(entity_type="CLIENT_NAME", term="Acme Corp", replace_with="CLIENT_01")
    ]

    with patch(
        "hey_jude.services.anonymizer.call_local_llm", new_callable=AsyncMock
    ) as mock_local, patch(
        "hey_jude.routes.safety_net_check", new_callable=AsyncMock
    ) as mock_safety, patch(
        "hey_jude.routes.route_completion", new_callable=AsyncMock
    ) as mock_route:
        # LLM finds nothing new; the seed alone must still anonymize the client.
        mock_local.return_value = json.dumps({
            "entities_found": [],
            "context_descriptors": {},
            "sensitivity": "low",
        })
        mock_safety.return_value = SafetyNetResult(
            passed=True, leaked_entities=[], auto_replaced=0
        )
        mock_route.return_value = {
            "id": "chatcmpl-known-llm",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "CLIENT_01 noted."},
                "finish_reason": "stop",
            }],
        }

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Acme Corp is our client."}],
            },
            headers={"X-API-Key": "sk-test-key"},
        )

    assert resp.status_code == 200
    sent = mock_route.await_args.kwargs["messages"][0].content
    assert "Acme Corp" not in sent
    assert "CLIENT_01" in sent
    assert resp.json()["choices"][0]["message"]["content"] == "Acme Corp noted."


def _audit_settings(tmp_path, **overrides):
    overrides.setdefault("anonymization_mode", "mechanical")
    return Settings(
        redis_url="redis://localhost:6379/1",
        redis_ttl_seconds=60,
        api_key="sk-test-key",
        audit_enabled=True,
        audit_destination=str(tmp_path / "audit.jsonl"),
        audit_rotation="none",
        **overrides,
    )


async def _audit_client(app):
    async with app.router.lifespan_context(app):
        app.state.redis_client = MemoryRedis()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_audit_record_written_on_success(tmp_path):
    settings = _audit_settings(tmp_path)
    app = create_app(settings)
    agen = _audit_client(app)
    client = await agen.__anext__()
    try:
        with patch(
            "hey_jude.routes.detect_entities", new_callable=AsyncMock
        ) as mock_detect, patch(
            "hey_jude.routes.substitute_entities", new_callable=AsyncMock
        ) as mock_substitute, patch(
            "hey_jude.routes.route_completion", new_callable=AsyncMock
        ) as mock_route:
            mock_detect.return_value = []
            mock_substitute.return_value = SubstitutionResult(
                mapping={},
                reverse_mapping={},
                context_descriptors={},
                sanitized_messages=[ChatMessage(role="user", content="hi")],
                sensitivity="low",
                needs_clarification=False,
                clarification_question=None,
            )
            mock_route.return_value = {
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}
                ]
            }
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o",
                      "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-API-Key": "sk-test-key", "X-Heyjude-Matter-Id": "M-42"},
            )
            assert resp.status_code == 200
    finally:
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()

    from hey_jude.services.audit import verify_chain
    path = str(tmp_path / "audit.jsonl")
    assert verify_chain(path).ok
    with open(path) as handle:
        record = json.loads(handle.readline())
    assert record["status"] == "completed"
    assert record["route"] == "chat_completions"
    assert record["matter_id"] == "M-42"
    assert record["total_ms"] is not None
    # metadata tier: hashes present, raw content absent
    assert record["input_sha256"]
    assert record["input"] is None


async def test_audit_record_written_on_auth_failure(tmp_path):
    settings = _audit_settings(tmp_path)
    app = create_app(settings)
    agen = _audit_client(app)
    client = await agen.__anext__()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401
    finally:
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()

    # Auth fails before request_id is generated, so no record is emitted.
    assert not (tmp_path / "audit.jsonl").exists()


async def test_audit_record_written_on_pipeline_error(tmp_path):
    settings = _audit_settings(tmp_path)
    app = create_app(settings)
    agen = _audit_client(app)
    client = await agen.__anext__()
    try:
        with patch(
            "hey_jude.routes.detect_entities", new_callable=AsyncMock
        ) as mock_detect, patch(
            "hey_jude.routes.substitute_entities", new_callable=AsyncMock
        ) as mock_substitute:
            mock_detect.return_value = []
            mock_substitute.side_effect = ValueError("boom")
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o",
                      "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-API-Key": "sk-test-key"},
            )
            assert resp.status_code == 502
    finally:
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()

    from hey_jude.services.audit import verify_chain
    path = str(tmp_path / "audit.jsonl")
    assert verify_chain(path).ok
    with open(path) as handle:
        record = json.loads(handle.readline())
    assert record["status"] == "error"
    assert record["error"] == "Anonymization validation failed"


async def test_audit_records_per_entity_decisions_in_llm_mode(tmp_path):
    settings = _audit_settings(
        tmp_path, anonymization_mode="llm", safety_net_strictness="off"
    )
    app = create_app(settings)
    agen = _audit_client(app)
    client = await agen.__anext__()
    try:
        with patch(
            "hey_jude.routes.anonymize_messages", new_callable=AsyncMock
        ) as mock_anon, patch(
            "hey_jude.routes.route_completion", new_callable=AsyncMock
        ) as mock_route:
            mock_anon.return_value = AnonymizationResult(
                mapping={"Acme Corp": "COMPANY_01"},
                reverse_mapping={"COMPANY_01": "Acme Corp"},
                context_descriptors={},
                sanitized_messages=[ChatMessage(role="user", content="COMPANY_01 matter")],
                sensitivity="low",
                entities_found=[
                    FoundEntity(
                        text="Acme Corp",
                        entity_type="ORGANIZATION",
                        action="replace",
                        replacement="COMPANY_01",
                        reason="real company name",
                    ),
                    FoundEntity(
                        text="the Agreement",
                        entity_type="MISC",
                        action="keep",
                        replacement=None,
                        reason="generic defined term",
                    ),
                ],
            )
            mock_route.return_value = {
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}
                ]
            }
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o",
                      "messages": [{"role": "user", "content": "Acme Corp matter"}]},
                headers={"X-API-Key": "sk-test-key"},
            )
            assert resp.status_code == 200
    finally:
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()

    path = str(tmp_path / "audit.jsonl")
    with open(path) as handle:
        record = json.loads(handle.readline())
    # metadata tier: type/action/reason recorded, raw entity text withheld
    assert record["decisions"] == [
        {"entity_type": "ORGANIZATION", "action": "replace", "reason": "real company name"},
        {"entity_type": "MISC", "action": "keep", "reason": "generic defined term"},
    ]
    assert "Acme Corp" not in json.dumps(record["decisions"])
