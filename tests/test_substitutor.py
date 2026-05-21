import json
import pytest
from hey_jude.config import Settings
from hey_jude.models import ChatMessage, DetectedEntity
from hey_jude.services.substitutor import (
    _build_deterministic_replacements,
    _build_placeholder_replacements,
    _build_prompt,
    _call_local_llm,
    _parse_llm_response,
    substitute_entities,
)
from unittest.mock import AsyncMock, patch


def test_build_deterministic_replacements():
    entities = [
        DetectedEntity(text="john@acme.com", entity_type="EMAIL_ADDRESS", start=0, end=13, score=0.9),
        DetectedEntity(text="555-1234", entity_type="PHONE_NUMBER", start=20, end=28, score=0.9),
    ]
    strategies = {"EMAIL_ADDRESS": "deterministic", "PHONE_NUMBER": "deterministic"}
    mapping = _build_deterministic_replacements(entities, strategies)
    assert mapping["john@acme.com"] == "user_1@example.com"
    assert mapping["555-1234"] == "555-0101"


def test_build_deterministic_skips_llm_entities():
    entities = [
        DetectedEntity(text="Microsoft", entity_type="ORGANIZATION", start=0, end=9, score=0.9),
        DetectedEntity(text="john@acme.com", entity_type="EMAIL_ADDRESS", start=15, end=28, score=0.9),
    ]
    strategies = {"ORGANIZATION": "llm", "EMAIL_ADDRESS": "deterministic"}
    mapping = _build_deterministic_replacements(entities, strategies)
    assert "Microsoft" not in mapping
    assert "john@acme.com" in mapping


def test_build_placeholder_replacements():
    entities = [
        DetectedEntity(text="John Smith", entity_type="PERSON", start=0, end=10, score=0.9),
        DetectedEntity(text="Jane Doe", entity_type="PERSON", start=15, end=23, score=0.9),
        DetectedEntity(text="Acme Corp", entity_type="ORGANIZATION", start=30, end=39, score=0.9),
        DetectedEntity(text="John Smith", entity_type="PERSON", start=45, end=55, score=0.9),
    ]
    strategies = {"PERSON": "placeholder", "ORGANIZATION": "placeholder"}
    mapping = _build_placeholder_replacements(entities, strategies)
    assert mapping["John Smith"] == "PERSON_01"
    assert mapping["Jane Doe"] == "PERSON_02"
    assert mapping["Acme Corp"] == "COMPANY_01"
    assert len(mapping) == 3


def test_build_placeholder_skips_non_placeholder_entities():
    entities = [
        DetectedEntity(text="John Smith", entity_type="PERSON", start=0, end=10, score=0.9),
        DetectedEntity(text="john@acme.com", entity_type="EMAIL_ADDRESS", start=15, end=28, score=0.9),
        DetectedEntity(text="Acme Corp", entity_type="ORGANIZATION", start=35, end=44, score=0.9),
    ]
    strategies = {
        "PERSON": "placeholder",
        "ORGANIZATION": "placeholder",
        "EMAIL_ADDRESS": "deterministic",
    }
    mapping = _build_placeholder_replacements(entities, strategies)
    assert "John Smith" in mapping
    assert "Acme Corp" in mapping
    assert "john@acme.com" not in mapping


def test_build_prompt():
    entities = [
        DetectedEntity(text="Meta", entity_type="ORGANIZATION", start=0, end=4, score=0.9),
        DetectedEntity(text="Microsoft", entity_type="ORGANIZATION", start=8, end=17, score=0.9),
    ]
    query = "Meta vs Microsoft IP case"
    prompt = _build_prompt(entities, query)
    assert "Meta" in prompt
    assert "Microsoft" in prompt
    assert "ORGANIZATION" in prompt
    assert "sensitivity" in prompt


def test_parse_llm_response_valid():
    raw = json.dumps({
        "sensitivity": "high",
        "reasoning": "test",
        "mapping": {"Meta": "Vertex Holdings"},
        "context_descriptors": {"Vertex Holdings": "social media company"},
        "sanitized_text": "Vertex Holdings did a thing",
        "needs_clarification": False,
        "clarification_question": None,
    })
    result = _parse_llm_response(raw)
    assert result["sensitivity"] == "high"
    assert result["mapping"]["Meta"] == "Vertex Holdings"


def test_parse_llm_response_invalid():
    with pytest.raises(ValueError, match="Failed to parse"):
        _parse_llm_response("this is not json at all")


def test_parse_llm_response_json_in_markdown():
    raw = '```json\n{"sensitivity": "low", "reasoning": "", "mapping": {}, "context_descriptors": {}, "sanitized_text": "hello", "needs_clarification": false, "clarification_question": null}\n```'
    result = _parse_llm_response(raw)
    assert result["sensitivity"] == "low"


async def test_call_local_llm_uses_ollama_native_api_with_thinking_disabled(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={
            "message": {
                "role": "assistant",
                "content": '{"sensitivity":"low","mapping":{},"context_descriptors":{},"needs_clarification":false,"clarification_question":null}',
            }
        },
    )

    raw = await _call_local_llm(
        "Return JSON only.",
        Settings(local_llm_url="http://localhost:11434/v1"),
    )

    request = httpx_mock.get_request()
    assert request is not None
    payload = json.loads(request.content)
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["temperature"] == 0.3
    assert payload["options"]["num_predict"] == 1024
    assert raw.startswith('{"sensitivity":"low"')


async def test_substitute_entities_ignores_llm_sanitized_text_by_default():
    raw = json.dumps({
        "sensitivity": "high",
        "reasoning": "Specific named relationship could identify the matter.",
        "mapping": {
            "Meta": "Vertex Holdings",
            "Microsoft": "Pinnacle Systems",
            "Instagram": "Photogram",
        },
        "context_descriptors": {
            "Vertex Holdings": "large social media conglomerate",
            "Pinnacle Systems": "major technology corporation",
            "Photogram": "social platform",
        },
        "sanitized_text": "Find analogous IP disputes between two technology companies involving a social media product.",
        "needs_clarification": False,
        "clarification_question": None,
    })
    messages = [
        ChatMessage(
            role="user",
            content="Bring up Meta vs Microsoft IP cases around Instagram",
        )
    ]
    entities = [
        DetectedEntity(text="Meta", entity_type="ORGANIZATION", start=9, end=13, score=0.9),
        DetectedEntity(text="Microsoft", entity_type="ORGANIZATION", start=17, end=26, score=0.9),
        DetectedEntity(text="Instagram", entity_type="ORGANIZATION", start=43, end=52, score=0.9),
    ]

    with patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local_llm:
        mock_local_llm.return_value = raw

        result = await substitute_entities(messages, entities, Settings())

    assert result.sanitized_messages[0].content == (
        "Bring up Vertex Holdings vs Pinnacle Systems IP cases around Photogram"
    )


async def test_substitute_entities_preserves_json_shape_while_replacing_values():
    raw = json.dumps({
        "sensitivity": "low",
        "reasoning": "Entity replacement only.",
        "mapping": {
            "John Smith": "Richard Roe",
            "Microsoft": "Pinnacle Systems",
        },
        "context_descriptors": {
            "Richard Roe": "a legal professional",
            "Pinnacle Systems": "a technology company",
        },
        "sanitized_text": "Ignore this rewrite entirely.",
        "needs_clarification": False,
        "clarification_question": None,
    })
    messages = [
        ChatMessage(
            role="user",
            content=json.dumps({
                "task": "summarize",
                "client": "John Smith",
                "target": "Microsoft",
                "schema": {"risk": "string", "next_steps": ["string"]},
            }),
        )
    ]
    entities = [
        DetectedEntity(text="John Smith", entity_type="PERSON", start=32, end=42, score=0.9),
        DetectedEntity(text="Microsoft", entity_type="ORGANIZATION", start=55, end=64, score=0.9),
    ]

    with patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local_llm:
        mock_local_llm.return_value = raw

        result = await substitute_entities(messages, entities, Settings())

    payload = json.loads(result.sanitized_messages[0].content)
    assert payload == {
        "task": "summarize",
        "client": "Richard Roe",
        "target": "Pinnacle Systems",
        "schema": {"risk": "string", "next_steps": ["string"]},
    }


async def test_substitute_entities_sanitizes_context_descriptor_keys_and_values():
    raw = json.dumps({
        "sensitivity": "high",
        "reasoning": "Named relationship could identify the person.",
        "mapping": {
            "Nick Watson": "Alex Mercer",
            "Google": "TechCorp",
        },
        "context_descriptors": {
            "Nick Watson": "Nick Watson is an employee at Google",
            "Google": "Google is the employer",
        },
        "sanitized_text": "",
        "needs_clarification": False,
        "clarification_question": None,
    })
    messages = [
        ChatMessage(role="user", content="Hello, I'm Nick Watson and work at Google")
    ]
    entities = [
        DetectedEntity(text="Nick Watson", entity_type="PERSON", start=10, end=21, score=0.9),
        DetectedEntity(text="Google", entity_type="ORGANIZATION", start=34, end=40, score=0.9),
    ]

    with patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local_llm:
        mock_local_llm.return_value = raw

        result = await substitute_entities(messages, entities, Settings())

    descriptor_text = json.dumps(result.context_descriptors)
    assert "Nick Watson" not in descriptor_text
    assert "Google" not in descriptor_text
    assert result.context_descriptors == {
        "Alex Mercer": "Alex Mercer is an employee at TechCorp",
        "TechCorp": "TechCorp is the employer",
    }


async def test_substitute_entities_accepts_list_context_descriptors_from_local_llm():
    raw = json.dumps({
        "sensitivity": "low",
        "reasoning": "Entity replacement only.",
        "mapping": {
            "Nick Watson": "Alex Mercer",
            "Google": "TechNova Inc.",
        },
        "context_descriptors": [
            {
                "entity": "Alex Mercer",
                "description": "individual professional",
            },
            {
                "entity": "TechNova Inc.",
                "description": "technology company",
            },
        ],
        "sanitized_text": "",
        "needs_clarification": False,
        "clarification_question": None,
    })
    messages = [
        ChatMessage(role="user", content="Hello, I'm Nick Watson and work at Google")
    ]
    entities = [
        DetectedEntity(text="Nick Watson", entity_type="PERSON", start=11, end=22, score=0.9),
        DetectedEntity(text="Google", entity_type="ORGANIZATION", start=35, end=41, score=0.9),
    ]

    with patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local_llm:
        mock_local_llm.return_value = raw

        result = await substitute_entities(messages, entities, Settings())

    assert result.context_descriptors == {
        "Alex Mercer": "individual professional",
        "TechNova Inc.": "technology company",
    }
    assert result.sanitized_messages[0].content == (
        "Hello, I'm Alex Mercer and work at TechNova Inc."
    )


async def test_substitute_entities_retries_invalid_llm_mapping_contract():
    invalid = json.dumps({
        "sensitivity": "low",
        "reasoning": "Only mapped one entity.",
        "mapping": {"Meta": "Vertex Holdings"},
        "context_descriptors": {},
        "sanitized_text": "",
        "needs_clarification": False,
        "clarification_question": None,
    })
    valid = json.dumps({
        "sensitivity": "low",
        "reasoning": "All entities mapped.",
        "mapping": {
            "Meta": "Vertex Holdings",
            "Microsoft": "Pinnacle Systems",
        },
        "context_descriptors": {},
        "sanitized_text": "",
        "needs_clarification": False,
        "clarification_question": None,
    })
    messages = [
        ChatMessage(role="user", content="Compare Meta and Microsoft")
    ]
    entities = [
        DetectedEntity(text="Meta", entity_type="ORGANIZATION", start=8, end=12, score=0.9),
        DetectedEntity(text="Microsoft", entity_type="ORGANIZATION", start=17, end=26, score=0.9),
    ]

    with patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local_llm:
        mock_local_llm.side_effect = [invalid, valid]

        result = await substitute_entities(messages, entities, Settings())

    assert mock_local_llm.await_count == 2
    assert result.mapping == {
        "Meta": "Vertex Holdings",
        "Microsoft": "Pinnacle Systems",
    }
    assert result.sanitized_messages[0].content == (
        "Compare Vertex Holdings and Pinnacle Systems"
    )


async def test_substitute_entities_rejects_mapping_that_leaks_original_text_after_retry():
    leaking = json.dumps({
        "sensitivity": "low",
        "reasoning": "Bad replacement leaks the original.",
        "mapping": {"Microsoft": "Microsoft Corporation"},
        "context_descriptors": {},
        "sanitized_text": "",
        "needs_clarification": False,
        "clarification_question": None,
    })
    messages = [
        ChatMessage(role="user", content="Summarize Microsoft licensing risk")
    ]
    entities = [
        DetectedEntity(text="Microsoft", entity_type="ORGANIZATION", start=10, end=19, score=0.9),
    ]

    with patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local_llm:
        mock_local_llm.return_value = leaking

        with pytest.raises(ValueError, match="Local LLM returned invalid anonymization contract"):
            await substitute_entities(messages, entities, Settings())

    assert mock_local_llm.await_count == 2


async def test_substitute_entities_rejects_json_shape_mismatch_before_routing():
    raw = json.dumps({
        "sensitivity": "low",
        "reasoning": "Entity replacement only.",
        "mapping": {"Microsoft": "Pinnacle Systems"},
        "context_descriptors": {},
        "sanitized_text": "",
        "needs_clarification": False,
        "clarification_question": None,
    })
    messages = [
        ChatMessage(
            role="user",
            content=json.dumps({
                "task": "summarize",
                "target": "Microsoft",
                "schema": {"risk": "string"},
            }),
        )
    ]
    entities = [
        DetectedEntity(text="Microsoft", entity_type="ORGANIZATION", start=32, end=41, score=0.9),
    ]

    with patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local_llm, patch(
        "hey_jude.services.substitutor._apply_mapping_preserving_text_structure",
    ) as mock_apply_mapping:
        mock_local_llm.return_value = raw
        mock_apply_mapping.return_value = json.dumps({
            "task": "summarize",
            "target": "Pinnacle Systems",
        })

        with pytest.raises(ValueError, match="Sanitized message structure mismatch"):
            await substitute_entities(messages, entities, Settings())
