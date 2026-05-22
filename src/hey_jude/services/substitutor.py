import json
import re
from typing import Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hey_jude.config import Settings
from hey_jude.models import ChatMessage, DetectedEntity, SubstitutionResult
from hey_jude.services.text import (
    apply_mapping_to_text as _apply_mapping_to_text,
    apply_mapping_preserving_text_structure as _apply_mapping_preserving_text_structure,
)


class LocalAnonymizationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sensitivity: Literal["low", "high"]
    reasoning: str | None = None
    mapping: dict[str, str]
    context_descriptors: dict[str, str] = Field(default_factory=dict)
    sanitized_text: str | None = None
    needs_clarification: bool
    clarification_question: str | None = None


_PLACEHOLDER_LABELS = {
    "ORGANIZATION": "COMPANY",
}


def _build_placeholder_replacements(
    entities: list[DetectedEntity],
    strategies: dict[str, str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}
    for entity in entities:
        if strategies.get(entity.entity_type) != "placeholder":
            continue
        if entity.text in mapping:
            continue
        label = _PLACEHOLDER_LABELS.get(entity.entity_type, entity.entity_type)
        counters[label] = counters.get(label, 0) + 1
        mapping[entity.text] = f"{label}_{counters[label]:02d}"
    return mapping


def _build_deterministic_replacements(
    entities: list[DetectedEntity],
    strategies: dict[str, str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}
    for entity in entities:
        if strategies.get(entity.entity_type) != "deterministic":
            continue
        if entity.text in mapping:
            continue
        counters[entity.entity_type] = counters.get(entity.entity_type, 0) + 1
        counter = counters[entity.entity_type]
        if entity.entity_type == "EMAIL_ADDRESS":
            mapping[entity.text] = f"user_{counter}@example.com"
        elif entity.entity_type == "PHONE_NUMBER":
            mapping[entity.text] = f"555-{100 + counter:04d}"
        else:
            mapping[entity.text] = f"REDACTED_{entity.entity_type}_{counter}"
    return mapping


def _build_prompt(entities: list[DetectedEntity], query: str) -> str:
    entity_list = json.dumps(
        [{"text": e.text, "type": e.entity_type} for e in entities],
        indent=2,
    )
    return f"""<task>
Analyze this legal professional's query for sensitive entity handling.
</task>

<entities>
{entity_list}
</entities>

<query>
{query}
</query>

<instructions>
1. Classify sensitivity: "low" if entity replacement alone prevents identification, "high" if structural patterns or relationships could de-anonymize.
2. For each entity, provide a brief context descriptor (what kind of entity it is, without identifying it).
3. Generate fictional replacement names that preserve the entity's role and domain. Always use fictional names, never descriptive phrases.
4. If high sensitivity: also rephrase the query to obscure identifying structural patterns while preserving the legal question's intent.
5. If you are unsure about the appropriate anonymization strategy, set needs_clarification to true and provide a clarification question.
6. Return ONLY a JSON object with these keys: sensitivity, reasoning, mapping, context_descriptors, sanitized_text, needs_clarification, clarification_question
</instructions>"""


def _parse_llm_response(raw: str) -> dict:
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse local LLM response as JSON: {e}") from e


def _normalize_context_descriptors(parsed: dict) -> dict:
    descriptors = parsed.get("context_descriptors")
    if isinstance(descriptors, dict):
        return parsed
    if not isinstance(descriptors, list):
        return parsed

    normalized = {}
    for item in descriptors:
        if not isinstance(item, dict):
            continue
        key = item.get("entity") or item.get("name") or item.get("text")
        value = (
            item.get("description")
            or item.get("descriptor")
            or item.get("context")
        )
        if isinstance(key, str) and isinstance(value, str):
            normalized[key] = value

    return {**parsed, "context_descriptors": normalized}


def _validate_llm_response(
    parsed: dict,
    entities: list[DetectedEntity],
) -> LocalAnonymizationResponse:
    try:
        result = LocalAnonymizationResponse.model_validate(
            _normalize_context_descriptors(parsed)
        )
    except ValidationError as e:
        raise ValueError(
            f"Local LLM returned invalid anonymization contract: {e}"
        ) from e

    if result.needs_clarification:
        return result

    missing = sorted(
        {
            entity.text
            for entity in entities
            if entity.text not in result.mapping
        }
    )
    if missing:
        raise ValueError(
            "Local LLM returned invalid anonymization contract: "
            f"missing mappings for {', '.join(missing)}"
        )

    for original, replacement in result.mapping.items():
        if original.casefold() in replacement.casefold():
            raise ValueError(
                "Local LLM returned invalid anonymization contract: "
                f"replacement for {original!r} leaks original text"
            )

    return result


def _build_retry_prompt(prompt: str, error: ValueError) -> str:
    return (
        prompt
        + "\n\nIMPORTANT: Your previous response was invalid. "
        + "Return ONLY a valid JSON object matching the requested schema. "
        + "Every listed entity must have a fictional replacement string. "
        + "Replacement strings must not contain the original sensitive text. "
        + f"Validation error: {error}"
    )


def _ollama_native_chat_url(local_llm_url: str) -> str | None:
    parsed = urlparse(local_llm_url)
    if parsed.port != 11434:
        return None

    path = parsed.path.rstrip("/")
    if path not in {"", "/v1"}:
        return None

    return f"{parsed.scheme}://{parsed.netloc}/api/chat"


async def _call_ollama_native_llm(prompt: str, settings: Settings, url: str) -> str:
    payload = {
        "model": settings.local_llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        },
    }
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


async def _call_local_llm(prompt: str, settings: Settings) -> str:
    ollama_url = _ollama_native_chat_url(settings.local_llm_url)
    if ollama_url:
        return await _call_ollama_native_llm(prompt, settings, ollama_url)

    url = f"{settings.local_llm_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.local_llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _sanitize_context_descriptors(
    context_descriptors: dict[str, str],
    mapping: dict[str, str],
) -> dict[str, str]:
    return {
        _apply_mapping_to_text(key, mapping): _apply_mapping_to_text(value, mapping)
        for key, value in context_descriptors.items()
    }


def _json_value_shape_matches(original: object, sanitized: object) -> bool:
    if isinstance(original, str):
        return isinstance(sanitized, str)
    if isinstance(original, list):
        return (
            isinstance(sanitized, list)
            and len(original) == len(sanitized)
            and all(
                _json_value_shape_matches(original_item, sanitized_item)
                for original_item, sanitized_item in zip(original, sanitized)
            )
        )
    if isinstance(original, dict):
        return (
            isinstance(sanitized, dict)
            and set(original.keys()) == set(sanitized.keys())
            and all(
                _json_value_shape_matches(original[key], sanitized[key])
                for key in original
            )
        )
    return original == sanitized


def _validate_sanitized_messages(
    original_messages: list[ChatMessage],
    sanitized_messages: list[ChatMessage],
) -> None:
    if len(original_messages) != len(sanitized_messages):
        raise ValueError("Sanitized message structure mismatch: message count changed")

    for original, sanitized in zip(original_messages, sanitized_messages):
        if original.role != sanitized.role:
            raise ValueError("Sanitized message structure mismatch: role changed")

        try:
            original_json = json.loads(original.content)
        except json.JSONDecodeError:
            continue

        if not isinstance(original_json, (dict, list)):
            continue

        try:
            sanitized_json = json.loads(sanitized.content)
        except json.JSONDecodeError as e:
            raise ValueError(
                "Sanitized message structure mismatch: JSON content became invalid"
            ) from e

        if not _json_value_shape_matches(original_json, sanitized_json):
            raise ValueError("Sanitized message structure mismatch")


async def substitute_entities(
    messages: list[ChatMessage],
    entities: list[DetectedEntity],
    settings: Settings,
    force_local_pass: bool = False,
) -> SubstitutionResult:
    placeholder_mapping = _build_placeholder_replacements(
        entities, settings.entity_strategies
    )

    deterministic_mapping = _build_deterministic_replacements(
        entities, settings.entity_strategies
    )

    llm_entities = [
        e for e in entities
        if settings.entity_strategies.get(e.entity_type) == "llm"
    ]

    llm_mapping: dict[str, str] = {}
    context_descriptors: dict[str, str] = {}
    sensitivity = "low"
    needs_clarification = False
    clarification_question = None

    should_call_local = force_local_pass or bool(llm_entities)

    if should_call_local:
        user_text = " ".join(
            m.content for m in messages if m.role in ("user", "assistant")
        )
        if settings.max_context_window and len(user_text) > settings.max_context_window:
            user_text = user_text[:settings.max_context_window]

        prompt = _build_prompt(llm_entities, user_text)
        last_error: ValueError | None = None
        for attempt in range(2):
            llm_prompt = prompt if attempt == 0 else _build_retry_prompt(
                prompt, last_error or ValueError("invalid response")
            )
            raw_response = await _call_local_llm(llm_prompt, settings)

            try:
                parsed = _parse_llm_response(raw_response)
                validated = _validate_llm_response(parsed, llm_entities)
                break
            except ValueError as e:
                last_error = e
        else:
            raise last_error or ValueError(
                "Local LLM returned invalid anonymization contract"
            )

        llm_mapping = validated.mapping
        context_descriptors = validated.context_descriptors
        sensitivity = validated.sensitivity
        needs_clarification = validated.needs_clarification
        clarification_question = validated.clarification_question

        if settings.always_full_anonymization:
            sensitivity = "high"

    full_mapping = {**placeholder_mapping, **deterministic_mapping, **llm_mapping}
    reverse_mapping = {v: k for k, v in full_mapping.items()}
    context_descriptors = _sanitize_context_descriptors(
        context_descriptors,
        full_mapping,
    )

    sanitized_messages = []
    for msg in messages:
        if msg.role == "system" and settings.passthrough_system_messages:
            sanitized_messages.append(msg)
        else:
            sanitized_content = _apply_mapping_preserving_text_structure(
                msg.content, full_mapping
            )
            sanitized_messages.append(
                ChatMessage(role=msg.role, content=sanitized_content)
            )

    _validate_sanitized_messages(messages, sanitized_messages)

    return SubstitutionResult(
        mapping=full_mapping,
        reverse_mapping=reverse_mapping,
        context_descriptors=context_descriptors,
        sanitized_messages=sanitized_messages,
        sensitivity=sensitivity,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
    )
