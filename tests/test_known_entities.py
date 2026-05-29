import json

import pytest

from hey_jude.config import Settings
from hey_jude.models import ChatMessage
from hey_jude.services.known_entities import (
    KnownEntity,
    load_known_entities,
    seed_mapping,
)


@pytest.fixture
def settings():
    return Settings()


def _msg(content):
    return [ChatMessage(role="user", content=content)]


def test_load_known_entities_yaml(tmp_path):
    path = tmp_path / "known.yaml"
    path.write_text(
        "- entity_type: CLIENT_NAME\n"
        "  term: Acme Corporation\n"
        "  aliases: [Acme, ACM]\n"
        "  replace_with: CLIENT_01\n"
    )
    known = load_known_entities(str(path))
    assert known == [
        KnownEntity(
            entity_type="CLIENT_NAME",
            term="Acme Corporation",
            aliases=("Acme", "ACM"),
            replace_with="CLIENT_01",
        )
    ]


def test_load_known_entities_json(tmp_path):
    path = tmp_path / "known.json"
    path.write_text(
        json.dumps([{"entity_type": "PERSONNEL", "term": "Jane Doe"}])
    )
    known = load_known_entities(str(path))
    assert known[0].term == "Jane Doe"
    assert known[0].aliases == ()
    assert known[0].replace_with is None


def test_load_known_entities_missing_term_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- entity_type: CLIENT_NAME\n  aliases: [Acme]\n")
    with pytest.raises(ValueError, match="non-empty 'term'"):
        load_known_entities(str(path))


def test_load_known_entities_not_a_list_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("entity_type: CLIENT_NAME\nterm: Acme\n")
    with pytest.raises(ValueError, match="must be a list"):
        load_known_entities(str(path))


def test_load_known_entities_unsupported_extension(tmp_path):
    path = tmp_path / "known.txt"
    path.write_text("nope")
    with pytest.raises(ValueError, match="unsupported extension"):
        load_known_entities(str(path))


def test_seed_mapping_groups_aliases_to_one_placeholder(settings):
    known = [
        KnownEntity(
            entity_type="CLIENT_NAME",
            term="Acme Corporation",
            aliases=("Acme", "ACM"),
        )
    ]
    mapping = seed_mapping(_msg("Acme Corporation, aka Acme, aka ACM"), known, settings)
    assert set(mapping.values()) == {"CLIENT_NAME_01"}
    assert mapping["Acme Corporation"] == "CLIENT_NAME_01"
    assert mapping["Acme"] == "CLIENT_NAME_01"
    assert mapping["ACM"] == "CLIENT_NAME_01"


def test_seed_mapping_honors_replace_with(settings):
    known = [
        KnownEntity(
            entity_type="CLIENT_NAME",
            term="Acme",
            replace_with="CLIENT_01",
        )
    ]
    mapping = seed_mapping(_msg("We act for Acme."), known, settings)
    assert mapping == {"Acme": "CLIENT_01"}


def test_seed_mapping_case_insensitive_captures_real_surface(settings):
    known = [KnownEntity(entity_type="CLIENT_NAME", term="Acme")]
    mapping = seed_mapping(_msg("call with acme and ACME today"), known, settings)
    # Both real surface forms are captured so replacement is guaranteed to apply.
    assert mapping["acme"] == "CLIENT_NAME_01"
    assert mapping["ACME"] == "CLIENT_NAME_01"


def test_seed_mapping_word_boundary_no_substring_hit(settings):
    known = [KnownEntity(entity_type="CLIENT_NAME", term="Acme")]
    mapping = seed_mapping(_msg("The Acmemeter is unrelated."), known, settings)
    assert mapping == {}


def test_seed_mapping_skips_absent_entities(settings):
    known = [
        KnownEntity(entity_type="CLIENT_NAME", term="Acme"),
        KnownEntity(entity_type="CLIENT_NAME", term="Globex"),
    ]
    mapping = seed_mapping(_msg("Only Globex is mentioned."), known, settings)
    # Acme absent, so the first assigned counter goes to Globex.
    assert mapping == {"Globex": "CLIENT_NAME_01"}


def test_seed_mapping_numbers_per_entity_type(settings):
    known = [
        KnownEntity(entity_type="CLIENT_NAME", term="Acme"),
        KnownEntity(entity_type="CLIENT_NAME", term="Globex"),
        KnownEntity(entity_type="PERSONNEL", term="Jane Doe"),
    ]
    mapping = seed_mapping(_msg("Acme, Globex and Jane Doe met."), known, settings)
    assert mapping["Acme"] == "CLIENT_NAME_01"
    assert mapping["Globex"] == "CLIENT_NAME_02"
    assert mapping["Jane Doe"] == "PERSONNEL_01"


def test_seed_mapping_empty_without_known(settings):
    assert seed_mapping(_msg("Acme"), [], settings) == {}


def test_seed_mapping_respects_passthrough_system_messages():
    settings = Settings(passthrough_system_messages=True)
    known = [KnownEntity(entity_type="CLIENT_NAME", term="Acme")]
    messages = [
        ChatMessage(role="system", content="Acme is the client."),
        ChatMessage(role="user", content="What is the weather?"),
    ]
    # System message is passed through untouched, so it must not be scanned.
    assert seed_mapping(messages, known, settings) == {}
