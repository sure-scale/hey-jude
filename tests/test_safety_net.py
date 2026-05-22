import pytest
from unittest.mock import AsyncMock, patch

from hey_jude.config import Settings
from hey_jude.models import ChatMessage, DetectedEntity
from hey_jude.services.safety_net import safety_net_check


async def test_safety_net_passes_when_no_entities_detected():
    messages = [ChatMessage(role="user", content="SOFTWARE_COMPANY_01 filed a claim")]
    mapping = {"Microsoft": "SOFTWARE_COMPANY_01"}
    settings = Settings(safety_net_strictness="warn")

    with patch(
        "hey_jude.services.safety_net.detect_entities",
        new_callable=AsyncMock,
    ) as mock_detect:
        mock_detect.return_value = []
        result = await safety_net_check(messages, mapping, settings)

    assert result.passed is True
    assert result.leaked_entities == []
    assert result.auto_replaced == 0


async def test_safety_net_filters_own_placeholders():
    messages = [ChatMessage(role="user", content="PERSON_01 works at SOFTWARE_COMPANY_01")]
    mapping = {"John": "PERSON_01", "Microsoft": "SOFTWARE_COMPANY_01"}
    settings = Settings(safety_net_strictness="warn")

    with patch(
        "hey_jude.services.safety_net.detect_entities",
        new_callable=AsyncMock,
    ) as mock_detect:
        mock_detect.return_value = [
            DetectedEntity(text="PERSON_01", entity_type="PERSON", start=0, end=9, score=0.5),
        ]
        result = await safety_net_check(messages, mapping, settings)

    assert result.passed is True
    assert result.leaked_entities == []


async def test_safety_net_filters_placeholder_pattern():
    messages = [ChatMessage(role="user", content="INVESTMENT_BANK_01 did a thing")]
    mapping = {"Goldman Sachs": "INVESTMENT_BANK_01"}
    settings = Settings(safety_net_strictness="warn")

    with patch(
        "hey_jude.services.safety_net.detect_entities",
        new_callable=AsyncMock,
    ) as mock_detect:
        mock_detect.return_value = [
            DetectedEntity(text="INVESTMENT_BANK_01", entity_type="ORGANIZATION",
                          start=0, end=18, score=0.4),
        ]
        result = await safety_net_check(messages, mapping, settings)

    assert result.passed is True


async def test_safety_net_warn_mode_auto_replaces():
    messages = [ChatMessage(role="user", content="John Smith works at SOFTWARE_COMPANY_01")]
    mapping = {"Microsoft": "SOFTWARE_COMPANY_01"}
    settings = Settings(safety_net_strictness="warn")

    with patch(
        "hey_jude.services.safety_net.detect_entities",
        new_callable=AsyncMock,
    ) as mock_detect:
        mock_detect.return_value = [
            DetectedEntity(text="John Smith", entity_type="PERSON",
                          start=0, end=10, score=0.9),
        ]
        result = await safety_net_check(messages, mapping, settings)

    assert result.passed is True
    assert result.auto_replaced == 1
    assert "MISSED_PERSON_01" in mapping["John Smith"]


async def test_safety_net_strict_mode_fails():
    messages = [ChatMessage(role="user", content="John Smith works at SOFTWARE_COMPANY_01")]
    mapping = {"Microsoft": "SOFTWARE_COMPANY_01"}
    settings = Settings(safety_net_strictness="strict")

    with patch(
        "hey_jude.services.safety_net.detect_entities",
        new_callable=AsyncMock,
    ) as mock_detect:
        mock_detect.return_value = [
            DetectedEntity(text="John Smith", entity_type="PERSON",
                          start=0, end=10, score=0.9),
        ]
        result = await safety_net_check(messages, mapping, settings)

    assert result.passed is False
    assert len(result.leaked_entities) == 1
    assert result.leaked_entities[0].text == "John Smith"


async def test_safety_net_off_mode_skips():
    messages = [ChatMessage(role="user", content="John Smith works here")]
    mapping = {}
    settings = Settings(safety_net_strictness="off")

    result = await safety_net_check(messages, mapping, settings)

    assert result.passed is True
    assert result.leaked_entities == []
    assert result.auto_replaced == 0
