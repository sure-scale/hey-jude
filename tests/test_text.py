import json
from hey_jude.services.text import (
    apply_mapping_to_text,
    apply_mapping_preserving_text_structure,
    replace_json_string_values,
)


def test_apply_mapping_to_text_replaces_longest_first():
    mapping = {"John": "PERSON_01", "John Smith": "PERSON_02"}
    result = apply_mapping_to_text("John Smith called John", mapping)
    assert result == "PERSON_02 called PERSON_01"


def test_apply_mapping_to_text_empty_mapping():
    assert apply_mapping_to_text("hello world", {}) == "hello world"


def test_replace_json_string_values_nested():
    value = {"a": "John", "b": ["John", {"c": "John"}]}
    mapping = {"John": "PERSON_01"}
    result = replace_json_string_values(value, mapping)
    assert result == {"a": "PERSON_01", "b": ["PERSON_01", {"c": "PERSON_01"}]}


def test_replace_json_string_values_non_string():
    assert replace_json_string_values(42, {"x": "y"}) == 42


def test_apply_mapping_preserving_text_structure_plain_text():
    result = apply_mapping_preserving_text_structure(
        "John works here", {"John": "PERSON_01"}
    )
    assert result == "PERSON_01 works here"


def test_apply_mapping_to_text_respects_word_boundaries():
    mapping = {"John": "PERSON_01"}
    result = apply_mapping_to_text("Johnson called John", mapping)
    assert result == "Johnson called PERSON_01"


def test_apply_mapping_to_text_no_replacement_chain():
    mapping = {"ABC": "DEF_01", "DEF": "GHI_01"}
    result = apply_mapping_to_text("ABC and DEF", mapping)
    assert result == "DEF_01 and GHI_01"


def test_apply_mapping_preserving_text_structure_json():
    original = json.dumps({"name": "John", "age": 30})
    result = apply_mapping_preserving_text_structure(original, {"John": "PERSON_01"})
    parsed = json.loads(result)
    assert parsed == {"name": "PERSON_01", "age": 30}
