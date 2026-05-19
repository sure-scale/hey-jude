import pytest
from hey_jude.config import Settings
from hey_jude.models import DetectedEntity
from hey_jude.services.detector import detect_entities


@pytest.fixture
def detector_settings():
    return Settings(
        presidio_entities=["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    )


async def test_detect_person_and_org(detector_settings):
    text = "John Smith works at Microsoft on the Azure platform."
    try:
        entities = await detect_entities(text, detector_settings)
    except OSError as e:
        if "can't find model" in str(e).lower() or "not find" in str(e).lower():
            pytest.skip("spacy model not installed")
        raise e
    entity_types = {e.entity_type for e in entities}
    entity_texts = {e.text for e in entities}
    assert "PERSON" in entity_types
    assert "ORGANIZATION" in entity_types
    assert "John Smith" in entity_texts


async def test_detect_email(detector_settings):
    text = "Contact us at john@example.com for details."
    try:
        entities = await detect_entities(text, detector_settings)
    except OSError as e:
        if "can't find model" in str(e).lower() or "not find" in str(e).lower():
            pytest.skip("spacy model not installed")
        raise e
    email_entities = [e for e in entities if e.entity_type == "EMAIL_ADDRESS"]
    assert len(email_entities) >= 1
    assert email_entities[0].text == "john@example.com"


async def test_detect_no_entities(detector_settings):
    text = "The weather is nice today."
    try:
        entities = await detect_entities(text, detector_settings)
    except OSError as e:
        if "can't find model" in str(e).lower() or "not find" in str(e).lower():
            pytest.skip("spacy model not installed")
        raise e
    assert entities == []


async def test_detected_entity_has_position(detector_settings):
    text = "John Smith is a lawyer."
    try:
        entities = await detect_entities(text, detector_settings)
    except OSError as e:
        if "can't find model" in str(e).lower() or "not find" in str(e).lower():
            pytest.skip("spacy model not installed")
        raise e
    person_entities = [e for e in entities if e.entity_type == "PERSON"]
    assert len(person_entities) >= 1
    entity = person_entities[0]
    assert entity.start >= 0
    assert entity.end > entity.start
    assert text[entity.start:entity.end] == entity.text
