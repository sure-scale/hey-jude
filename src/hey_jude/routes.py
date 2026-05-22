import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import httpx

from hey_jude.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    HeyJudeMetadata,
    SubstitutionResult,
)
from hey_jude.services.anonymizer import anonymize_messages
from hey_jude.services.detector import detect_entities
from hey_jude.services.mapper import reverse_map, reverse_map_text
from hey_jude.services.router import route_completion
from hey_jude.services.safety_net import safety_net_check
from hey_jude.services.substitutor import substitute_entities

router = APIRouter()


def _extract_api_key(request: Request) -> str | None:
    key = request.headers.get("X-API-Key")
    if key:
        return key
    key = request.headers.get("X-Goog-Api-Key")
    if key:
        return key
    key = request.query_params.get("key")
    if key:
        return key
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def _completion_kwargs(body: ChatCompletionRequest) -> dict:
    kwargs = {}
    if body.temperature is not None:
        kwargs["temperature"] = body.temperature
    if body.max_tokens is not None:
        kwargs["max_tokens"] = body.max_tokens
    if body.top_p is not None:
        kwargs["top_p"] = body.top_p
    if body.frequency_penalty is not None:
        kwargs["frequency_penalty"] = body.frequency_penalty
    if body.presence_penalty is not None:
        kwargs["presence_penalty"] = body.presence_penalty
    if body.response_format is not None:
        kwargs["response_format"] = body.response_format
    if body.tools is not None:
        kwargs["tools"] = body.tools
    if body.tool_choice is not None:
        kwargs["tool_choice"] = body.tool_choice
    return kwargs


def _route_kwargs(body: ChatCompletionRequest, settings) -> dict:
    kwargs = _completion_kwargs(body)
    if settings.external_llm_api_base:
        kwargs["api_base"] = settings.external_llm_api_base
    return kwargs


async def check_local_llm_health(settings) -> bool:
    url = f"{settings.local_llm_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            response = await client.get(url)
            return response.status_code < 500
    except Exception:
        return False


async def _messages_with_clarification_context(
    body: ChatCompletionRequest,
    redis_client,
) -> list:
    if not body.request_id:
        return body.messages

    stored_request = await redis_client.get_request(body.request_id)
    if stored_request is None:
        raise HTTPException(
            status_code=404,
            detail="Clarification request not found or expired",
        )

    original = ChatCompletionRequest.model_validate(stored_request)
    return [*original.messages, *body.messages]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return ""


def _choice_text(response: dict) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "") or ""


def _finish_reason(response: dict) -> str:
    choices = response.get("choices", [])
    if not choices:
        return "stop"
    return choices[0].get("finish_reason") or "stop"


def _usage(response: dict) -> dict:
    return response.get("usage") or {}


def _response_usage(response: dict) -> dict:
    usage = _usage(response)
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


async def _run_gateway_completion(
    body: ChatCompletionRequest,
    request: Request,
) -> dict | ChatCompletionResponse:
    settings = request.app.state.settings
    redis_client = request.app.state.redis_client

    api_key = _extract_api_key(request)
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    request_id = str(uuid.uuid4())
    messages = await _messages_with_clarification_context(body, redis_client)

    try:
        redis_ok = await redis_client.health_check()
        if not redis_ok:
            raise HTTPException(status_code=503, detail="Redis unavailable")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    if settings.anonymization_mode == "llm":
        prompt_template = getattr(request.app.state, "anonymization_prompt", None)

        try:
            anon_result = await anonymize_messages(
                messages, settings, prompt_template=prompt_template,
            )
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Local LLM unavailable")
        except ValueError:
            raise HTTPException(
                status_code=502, detail="Anonymization validation failed"
            )

        if settings.safety_net_strictness != "off":
            sn_result = await safety_net_check(
                anon_result.sanitized_messages, anon_result.mapping, settings,
            )
            if not sn_result.passed:
                raise HTTPException(
                    status_code=422,
                    detail="Safety net: PII detected in anonymized output",
                )

        sub_result = SubstitutionResult(
            mapping=anon_result.mapping,
            reverse_mapping={v: k for k, v in anon_result.mapping.items()},
            context_descriptors=anon_result.context_descriptors,
            sanitized_messages=anon_result.sanitized_messages,
            sensitivity=anon_result.sensitivity,
            needs_clarification=False,
            clarification_question=None,
        )
        all_entities_count = len([
            e for e in anon_result.entities_found if e.action != "keep"
        ])
    else:
        all_entities = []
        for msg in messages:
            if msg.role == "system" and settings.passthrough_system_messages:
                continue
            entities = await detect_entities(msg.content, settings)
            all_entities.extend(entities)

        try:
            sub_result = await substitute_entities(
                messages, all_entities, settings, force_local_pass=True
            )
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Local LLM unavailable")
        except ValueError:
            raise HTTPException(
                status_code=502, detail="Anonymization validation failed"
            )
        all_entities_count = len(all_entities)

    if sub_result.needs_clarification and settings.allow_clarification_requests:
        await redis_client.store_request(
            request_id,
            body.model_dump(exclude={"request_id"}),
            settings.redis_ttl_seconds,
        )
        return ChatCompletionResponse(
            id=f"heyjude-clarify-{request_id}",
            created=int(time.time()),
            model=body.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=sub_result.clarification_question
                        or "Could you clarify your request?",
                    ),
                    finish_reason="stop",
                )
            ],
            heyjude_metadata=HeyJudeMetadata(
                request_id=request_id,
                entities_detected=all_entities_count,
                sensitivity=sub_result.sensitivity,
                status="clarification_needed",
            ),
        )

    await redis_client.store_mapping(
        request_id,
        sub_result.mapping,
        settings.redis_ttl_seconds,
    )

    completion_kwargs = _route_kwargs(body, settings)

    external_response = await route_completion(
        messages=sub_result.sanitized_messages,
        context_descriptors=sub_result.context_descriptors,
        model=settings.external_llm_model,
        **completion_kwargs,
    )

    mapping = await redis_client.get_mapping(request_id)
    if mapping is None:
        raise HTTPException(
            status_code=504,
            detail="Redis mapping expired before response could be de-anonymized",
        )
    reverse_mapping = {v: k for k, v in mapping.items()}

    for choice in external_response.get("choices", []):
        content = choice.get("message", {}).get("content", "")
        if content:
            choice["message"]["content"] = reverse_map_text(content, reverse_mapping)

        for tool_call in choice.get("message", {}).get("tool_calls", []) or []:
            arguments = tool_call.get("function", {}).get("arguments", "")
            if arguments:
                tool_call["function"]["arguments"] = reverse_map_text(
                    arguments, reverse_mapping
                )

    external_response["heyjude_metadata"] = {
        "request_id": request_id,
        "entities_detected": all_entities_count,
        "sensitivity": sub_result.sensitivity,
        "status": "completed",
    }

    return external_response


def _anthropic_to_chat_request(body: dict) -> ChatCompletionRequest:
    messages = []
    system = body.get("system")
    if system:
        messages.append(ChatMessage(role="system", content=_content_to_text(system)))
    for message in body.get("messages", []):
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        messages.append(
            ChatMessage(role=role, content=_content_to_text(message.get("content", "")))
        )
    return ChatCompletionRequest(
        model=body["model"],
        messages=messages,
        request_id=body.get("request_id"),
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"),
        top_p=body.get("top_p"),
    )


def _chat_to_anthropic_response(response: dict | ChatCompletionResponse, model: str) -> dict:
    if isinstance(response, ChatCompletionResponse):
        response = response.model_dump()
    finish_reason = _finish_reason(response)
    stop_reason = "max_tokens" if finish_reason == "length" else "end_turn"
    usage = _usage(response)
    return {
        "id": response.get("id", f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": _choice_text(response)}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
        "heyjude_metadata": response.get("heyjude_metadata"),
    }


def _gemini_parts_to_text(parts: list) -> str:
    return "\n".join(
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


def _gemini_to_chat_request(model: str, body: dict) -> ChatCompletionRequest:
    messages = []
    system_instruction = body.get("systemInstruction")
    if isinstance(system_instruction, dict):
        system_text = _gemini_parts_to_text(system_instruction.get("parts", []))
        if system_text:
            messages.append(ChatMessage(role="system", content=system_text))

    contents = body.get("contents", [])
    if isinstance(contents, dict):
        contents = [contents]
    for content in contents:
        if not isinstance(content, dict):
            continue
        role = "assistant" if content.get("role") == "model" else "user"
        text = _gemini_parts_to_text(content.get("parts", []))
        messages.append(ChatMessage(role=role, content=text))

    generation_config = body.get("generationConfig") or {}
    return ChatCompletionRequest(
        model=model,
        messages=messages,
        request_id=body.get("request_id"),
        temperature=generation_config.get("temperature"),
        max_tokens=generation_config.get("maxOutputTokens"),
        top_p=generation_config.get("topP"),
    )


def _chat_to_gemini_response(response: dict | ChatCompletionResponse, model: str) -> dict:
    if isinstance(response, ChatCompletionResponse):
        response = response.model_dump()
    usage = _usage(response)
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": _choice_text(response)}],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        },
        "modelVersion": model,
        "heyjude_metadata": response.get("heyjude_metadata"),
    }


def _responses_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return ""


def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tools, list):
        return None

    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        if isinstance(tool.get("function"), dict):
            converted.append(tool)
            continue
        name = tool.get("name")
        if not isinstance(name, str):
            continue
        function = {"name": name}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        if isinstance(tool.get("parameters"), dict):
            function["parameters"] = tool["parameters"]
        converted.append({"type": "function", "function": function})
    return converted or None


def _responses_tool_choice_to_chat_tool_choice(tool_choice: Any) -> Any:
    if (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and isinstance(tool_choice.get("name"), str)
    ):
        return {
            "type": "function",
            "function": {"name": tool_choice["name"]},
        }
    return tool_choice


def _responses_to_chat_request(body: dict) -> ChatCompletionRequest:
    messages = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append(ChatMessage(role="system", content=instructions))

    input_value = body.get("input", "")
    if isinstance(input_value, str):
        messages.append(ChatMessage(role="user", content=input_value))
    elif isinstance(input_value, list):
        for item in input_value:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call_output":
                messages.append(
                    ChatMessage(
                        role="user",
                        content=_responses_content_to_text(item.get("output", "")),
                    )
                )
                continue

            role = item.get("role")
            if role not in {"system", "user", "assistant"}:
                continue
            messages.append(
                ChatMessage(
                    role=role,
                    content=_responses_content_to_text(item.get("content", "")),
                )
            )

    return ChatCompletionRequest(
        model=body["model"],
        messages=messages,
        request_id=body.get("request_id"),
        temperature=body.get("temperature"),
        max_tokens=body.get("max_output_tokens"),
        top_p=body.get("top_p"),
        frequency_penalty=body.get("frequency_penalty"),
        presence_penalty=body.get("presence_penalty"),
        tools=_responses_tools_to_chat_tools(body.get("tools")),
        tool_choice=_responses_tool_choice_to_chat_tool_choice(body.get("tool_choice")),
    )


def _chat_tool_calls(response: dict) -> list[dict[str, Any]]:
    choices = response.get("choices", [])
    if not choices:
        return []
    tool_calls = choices[0].get("message", {}).get("tool_calls")
    return tool_calls if isinstance(tool_calls, list) else []


def _responses_output_items(response: dict) -> list[dict[str, Any]]:
    output = []
    text = _choice_text(response)
    if text:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        )

    for tool_call in _chat_tool_calls(response):
        function = tool_call.get("function", {})
        output.append(
            {
                "id": tool_call.get("id", f"fc_{uuid.uuid4().hex}"),
                "type": "function_call",
                "status": "completed",
                "call_id": tool_call.get("id", f"call_{uuid.uuid4().hex}"),
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
            }
        )
    return output


def _chat_to_responses_response(
    response: dict | ChatCompletionResponse,
    model: str,
) -> dict:
    if isinstance(response, ChatCompletionResponse):
        response = response.model_dump()
    text = _choice_text(response)
    return {
        "id": response.get("id", f"resp_{uuid.uuid4().hex}"),
        "object": "response",
        "created_at": response.get("created", int(time.time())),
        "status": "completed",
        "model": model,
        "output": _responses_output_items(response),
        "output_text": text,
        "usage": _response_usage(response),
        "heyjude_metadata": response.get("heyjude_metadata"),
    }


async def _responses_event_generator(response: dict):
    response_id = response["id"]
    text = response.get("output_text", "")
    yield (
        "data: "
        + json.dumps(
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                    "output_text": "",
                },
            }
        )
        + "\n\n"
    )
    for item in response.get("output", []):
        if item.get("type") != "function_call":
            continue
        yield (
            "data: "
            + json.dumps({"type": "response.output_item.added", "item": item})
            + "\n\n"
        )
        yield (
            "data: "
            + json.dumps({"type": "response.output_item.done", "item": item})
            + "\n\n"
        )
    if text:
        yield (
            "data: "
            + json.dumps(
                {"type": "response.output_text.delta", "delta": text}
            )
            + "\n\n"
        )
    yield (
        "data: "
        + json.dumps({"type": "response.completed", "response": response})
        + "\n\n"
    )
    yield "data: [DONE]\n\n"


@router.get("/health")
async def health(request: Request):
    redis_ok = False
    try:
        redis_ok = await request.app.state.redis_client.health_check()
    except Exception:
        pass
    local_llm_ok = await check_local_llm_health(request.app.state.settings)
    return {
        "status": "ok" if redis_ok and local_llm_ok else "degraded",
        "redis": redis_ok,
        "local_llm": local_llm_ok,
    }


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    return await _run_gateway_completion(body, request)


@router.post("/v1/responses")
async def responses(body: dict, request: Request):
    chat_body = _responses_to_chat_request(body)
    response = await _run_gateway_completion(chat_body, request)
    responses_response = _chat_to_responses_response(response, body["model"])
    if body.get("stream"):
        return StreamingResponse(
            _responses_event_generator(responses_response),
            media_type="text/event-stream",
        )
    return responses_response


@router.post("/v1/messages")
async def anthropic_messages(body: dict, request: Request):
    chat_body = _anthropic_to_chat_request(body)
    response = await _run_gateway_completion(chat_body, request)
    if body.get("stream"):
        model = body["model"]
        text = _choice_text(response)
        
        async def event_generator():
            msg_id = f"msg_{uuid.uuid4().hex}"
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    return _chat_to_anthropic_response(response, body["model"])


@router.post("/v1beta/models/{model}:generateContent")
@router.post("/v1/models/{model}:generateContent")
async def gemini_generate_content(model: str, body: dict, request: Request):
    chat_body = _gemini_to_chat_request(model, body)
    response = await _run_gateway_completion(chat_body, request)
    return _chat_to_gemini_response(response, model)


@router.post("/v1beta/models/{model}:streamGenerateContent")
@router.post("/v1/models/{model}:streamGenerateContent")
async def gemini_stream_generate_content(model: str, body: dict, request: Request):
    chat_body = _gemini_to_chat_request(model, body)
    response = await _run_gateway_completion(chat_body, request)
    gemini_resp = _chat_to_gemini_response(response, model)
    
    async def event_generator():
        yield f"data: {json.dumps(gemini_resp)}\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")
