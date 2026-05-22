import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from hey_jude.config import Settings
from hey_jude.models import ChatMessage, FoundEntity, AnonymizationResult
from hey_jude.services.anonymizer import (
    validate_llm_anonymization_response,
    build_mapping_from_entities,
    render_prompt,
    anonymize_messages,
)


def test_validate_response_valid():
    raw = {
        "entities_found": [
            {"text": "Microsoft", "type": "ORGANIZATION", "action": "replace",
             "replacement": "SOFTWARE_COMPANY_01", "reason": "real company"},
            {"text": "Purchaser", "type": "DEFINED_TERM", "action": "keep",
             "replacement": None, "reason": "legal defined term"},
        ],
        "context_descriptors": {"SOFTWARE_COMPANY_01": "tech company"},
        "sensitivity": "low",
    }
    entities, descriptors, sensitivity = validate_llm_anonymization_response(raw)
    assert len(entities) == 2
    assert entities[0].action == "replace"
    assert entities[1].action == "keep"
    assert sensitivity == "low"


def test_validate_response_missing_replacement_for_replace():
    raw = {
        "entities_found": [
            {"text": "Microsoft", "type": "ORGANIZATION", "action": "replace",
             "replacement": None, "reason": "real company"},
        ],
        "context_descriptors": {},
        "sensitivity": "low",
    }
    with pytest.raises(ValueError, match="missing replacement"):
        validate_llm_anonymization_response(raw)


def test_validate_response_leak_detection():
    raw = {
        "entities_found": [
            {"text": "Microsoft", "type": "ORGANIZATION", "action": "replace",
             "replacement": "Microsoft Corp", "reason": "real company"},
        ],
        "context_descriptors": {},
        "sensitivity": "low",
    }
    with pytest.raises(ValueError, match="leaks original"):
        validate_llm_anonymization_response(raw)


def test_validate_response_duplicate_replacements():
    raw = {
        "entities_found": [
            {"text": "Microsoft", "type": "ORGANIZATION", "action": "replace",
             "replacement": "COMPANY_01", "reason": "company"},
            {"text": "Google", "type": "ORGANIZATION", "action": "replace",
             "replacement": "COMPANY_01", "reason": "company"},
        ],
        "context_descriptors": {},
        "sensitivity": "low",
    }
    with pytest.raises(ValueError, match="duplicate replacement"):
        validate_llm_anonymization_response(raw)


def test_build_mapping_from_entities():
    entities = [
        FoundEntity(text="Microsoft", entity_type="ORGANIZATION",
                    action="replace", replacement="SOFTWARE_COMPANY_01", reason=""),
        FoundEntity(text="Purchaser", entity_type="DEFINED_TERM",
                    action="keep", replacement=None, reason=""),
        FoundEntity(text="California", entity_type="LOCATION",
                    action="generalize", replacement="US_WEST_COAST", reason=""),
    ]
    mapping = build_mapping_from_entities(entities)
    assert mapping == {
        "Microsoft": "SOFTWARE_COMPANY_01",
        "California": "US_WEST_COAST",
    }
    assert "Purchaser" not in mapping


def test_render_prompt():
    template = Path("prompts/anonymize.txt").read_text()
    rendered = render_prompt(
        template,
        message_text="John works at Microsoft",
        existing_mapping={"Jane": "PERSON_01"},
    )
    assert "John works at Microsoft" in rendered
    assert '"Jane": "PERSON_01"' in rendered


def _make_llm_response(entities_found, context_descriptors=None, sensitivity="low"):
    return json.dumps({
        "entities_found": entities_found,
        "context_descriptors": context_descriptors or {},
        "sensitivity": sensitivity,
    })


async def test_anonymize_messages_single_message():
    llm_response = _make_llm_response([
        {"text": "Microsoft", "type": "ORGANIZATION", "action": "replace",
         "replacement": "SOFTWARE_COMPANY_01", "reason": "real company"},
        {"text": "Purchaser", "type": "DEFINED_TERM", "action": "keep",
         "replacement": None, "reason": "legal defined term"},
    ], {"SOFTWARE_COMPANY_01": "tech company"})

    messages = [
        ChatMessage(role="user", content="The Purchaser sued Microsoft for breach"),
    ]

    template = Path("prompts/anonymize.txt").read_text()
    settings = Settings(anonymization_mode="llm")

    with patch(
        "hey_jude.services.anonymizer.call_local_llm",
        new_callable=AsyncMock,
    ) as mock_llm:
        mock_llm.return_value = llm_response
        result = await anonymize_messages(messages, settings, prompt_template=template)

    assert result.mapping == {"Microsoft": "SOFTWARE_COMPANY_01"}
    assert result.sanitized_messages[0].content == (
        "The Purchaser sued SOFTWARE_COMPANY_01 for breach"
    )
    assert result.context_descriptors == {"SOFTWARE_COMPANY_01": "tech company"}
    assert len(result.entities_found) == 2


async def test_anonymize_messages_accumulates_mapping():
    response_1 = _make_llm_response([
        {"text": "Microsoft", "type": "ORGANIZATION", "action": "replace",
         "replacement": "SOFTWARE_COMPANY_01", "reason": ""},
    ])
    response_2 = _make_llm_response([
        {"text": "Microsoft", "type": "ORGANIZATION", "action": "replace",
         "replacement": "SOFTWARE_COMPANY_01", "reason": ""},
        {"text": "John Smith", "type": "PERSON", "action": "replace",
         "replacement": "PERSON_01", "reason": ""},
    ])

    messages = [
        ChatMessage(role="user", content="Microsoft filed a claim"),
        ChatMessage(role="user", content="John Smith represents Microsoft"),
    ]

    template = Path("prompts/anonymize.txt").read_text()
    settings = Settings(anonymization_mode="llm")

    with patch(
        "hey_jude.services.anonymizer.call_local_llm",
        new_callable=AsyncMock,
    ) as mock_llm:
        mock_llm.side_effect = [response_1, response_2]
        result = await anonymize_messages(messages, settings, prompt_template=template)

    assert result.mapping == {
        "Microsoft": "SOFTWARE_COMPANY_01",
        "John Smith": "PERSON_01",
    }
    assert result.sanitized_messages[0].content == "SOFTWARE_COMPANY_01 filed a claim"
    assert result.sanitized_messages[1].content == (
        "PERSON_01 represents SOFTWARE_COMPANY_01"
    )

    second_call_prompt = mock_llm.call_args_list[1].args[0]
    assert "SOFTWARE_COMPANY_01" in second_call_prompt


async def test_anonymize_messages_skips_system_when_passthrough():
    llm_response = _make_llm_response([])

    messages = [
        ChatMessage(role="system", content="You are a legal assistant."),
        ChatMessage(role="user", content="Hello"),
    ]

    template = Path("prompts/anonymize.txt").read_text()
    settings = Settings(anonymization_mode="llm", passthrough_system_messages=True)

    with patch(
        "hey_jude.services.anonymizer.call_local_llm",
        new_callable=AsyncMock,
    ) as mock_llm:
        mock_llm.return_value = llm_response
        result = await anonymize_messages(messages, settings, prompt_template=template)

    assert mock_llm.await_count == 1
    assert result.sanitized_messages[0].content == "You are a legal assistant."
    assert result.sanitized_messages[0].role == "system"


async def test_anonymize_messages_retries_on_validation_error():
    invalid = _make_llm_response([
        {"text": "Microsoft", "type": "ORGANIZATION", "action": "replace",
         "replacement": None, "reason": ""},
    ])
    valid = _make_llm_response([
        {"text": "Microsoft", "type": "ORGANIZATION", "action": "replace",
         "replacement": "SOFTWARE_COMPANY_01", "reason": ""},
    ])

    messages = [ChatMessage(role="user", content="Microsoft")]
    template = Path("prompts/anonymize.txt").read_text()
    settings = Settings(anonymization_mode="llm")

    with patch(
        "hey_jude.services.anonymizer.call_local_llm",
        new_callable=AsyncMock,
    ) as mock_llm:
        mock_llm.side_effect = [invalid, valid]
        result = await anonymize_messages(messages, settings, prompt_template=template)

    assert mock_llm.await_count == 2
    assert result.mapping == {"Microsoft": "SOFTWARE_COMPANY_01"}


async def test_anonymize_messages_with_existing_mapping():
    llm_response = _make_llm_response([
        {"text": "Microsoft", "type": "ORGANIZATION", "action": "replace",
         "replacement": "SOFTWARE_COMPANY_01", "reason": ""},
        {"text": "John Smith", "type": "PERSON", "action": "replace",
         "replacement": "PERSON_01", "reason": ""},
    ])

    messages = [
        ChatMessage(role="user", content="John Smith at Microsoft"),
    ]

    template = Path("prompts/anonymize.txt").read_text()
    settings = Settings(anonymization_mode="llm")
    existing = {"Microsoft": "SOFTWARE_COMPANY_01"}

    with patch(
        "hey_jude.services.anonymizer.call_local_llm",
        new_callable=AsyncMock,
    ) as mock_llm:
        mock_llm.return_value = llm_response
        result = await anonymize_messages(
            messages, settings, prompt_template=template, existing_mapping=existing,
        )

    assert result.mapping["Microsoft"] == "SOFTWARE_COMPANY_01"
    assert result.mapping["John Smith"] == "PERSON_01"
    prompt_sent = mock_llm.call_args.args[0]
    assert "SOFTWARE_COMPANY_01" in prompt_sent
