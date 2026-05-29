import pytest

from hey_jude.config import Settings
from hey_jude.services.detector import detect_entities
from hey_jude.services.recognizers import (
    PatternSpec,
    RecognizerSpec,
    build_recognizers,
    load_recognizer_specs,
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_load_recognizer_specs_yaml(tmp_path):
    path = _write(
        tmp_path,
        "rec.yaml",
        "- name: matter_number\n"
        "  entity_type: MATTER_NUMBER\n"
        "  patterns:\n"
        "    - regex: '\\bM-\\d{6}\\b'\n"
        "      score: 0.9\n"
        "  context: [matter, ref]\n",
    )
    specs = load_recognizer_specs(path)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.entity_type == "MATTER_NUMBER"
    assert spec.patterns[0].regex == r"\bM-\d{6}\b"
    assert spec.patterns[0].score == 0.9
    assert spec.context == ("matter", "ref")
    assert spec.strategy == "placeholder"


def test_load_recognizer_specs_default_score(tmp_path):
    path = _write(
        tmp_path,
        "rec.yaml",
        "- name: code\n  entity_type: CODE\n  patterns:\n    - regex: 'X\\d+'\n",
    )
    specs = load_recognizer_specs(path)
    assert specs[0].patterns[0].score == 0.85


def test_load_recognizer_missing_patterns_raises(tmp_path):
    path = _write(tmp_path, "rec.yaml", "- name: code\n  entity_type: CODE\n")
    with pytest.raises(ValueError, match="non-empty 'patterns'"):
        load_recognizer_specs(path)


def test_load_recognizer_bad_score_raises(tmp_path):
    path = _write(
        tmp_path,
        "rec.yaml",
        "- name: code\n  entity_type: CODE\n  patterns:\n    - regex: 'X'\n      score: 2\n",
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_recognizer_specs(path)


def test_load_recognizer_not_a_list_raises(tmp_path):
    path = _write(tmp_path, "rec.yaml", "name: code\n")
    with pytest.raises(ValueError, match="must be a list"):
        load_recognizer_specs(path)


def test_settings_merges_custom_recognizer_types(tmp_path):
    path = _write(
        tmp_path,
        "rec.yaml",
        "- name: matter_number\n"
        "  entity_type: MATTER_NUMBER\n"
        "  patterns:\n    - regex: '\\bM-\\d{6}\\b'\n      score: 0.9\n"
        "  strategy: deterministic\n",
    )
    settings = Settings(custom_recognizers_path=path)
    assert "MATTER_NUMBER" in settings.presidio_entities
    assert settings.entity_strategies["MATTER_NUMBER"] == "deterministic"
    assert len(settings.custom_recognizer_specs) == 1


def test_settings_does_not_override_existing_strategy(tmp_path):
    path = _write(
        tmp_path,
        "rec.yaml",
        "- name: person_code\n"
        "  entity_type: PERSON\n"
        "  patterns:\n    - regex: 'P\\d+'\n      score: 0.9\n"
        "  strategy: deterministic\n",
    )
    settings = Settings(custom_recognizers_path=path)
    # PERSON already maps to "placeholder" by default; custom spec must not clobber it.
    assert settings.entity_strategies["PERSON"] == "placeholder"


def test_build_recognizers_produces_pattern_recognizer():
    specs = [
        RecognizerSpec(
            name="matter_number",
            entity_type="MATTER_NUMBER",
            patterns=(PatternSpec(regex=r"\bM-\d{6}\b", score=0.9),),
        )
    ]
    recognizers = build_recognizers(specs)
    assert len(recognizers) == 1
    assert recognizers[0].supported_entities == ["MATTER_NUMBER"]


async def test_custom_recognizer_detects_entity(tmp_path):
    path = _write(
        tmp_path,
        "rec.yaml",
        "- name: matter_number\n"
        "  entity_type: MATTER_NUMBER\n"
        "  patterns:\n    - regex: '\\bM-\\d{6}\\b'\n      score: 0.9\n",
    )
    settings = Settings(custom_recognizers_path=path)
    try:
        entities = await detect_entities("Re: matter M-123456 filed today.", settings)
    except OSError as e:
        if "find model" in str(e).lower():
            pytest.skip("spacy model not installed")
        raise

    matter = [e for e in entities if e.entity_type == "MATTER_NUMBER"]
    assert len(matter) == 1
    assert matter[0].text == "M-123456"
