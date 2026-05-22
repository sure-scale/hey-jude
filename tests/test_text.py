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


def test_apply_mapping_to_text_email_address():
    mapping = {"john@acme.com": "person_01@company_01.com"}
    result = apply_mapping_to_text("Contact john@acme.com for details", mapping)
    assert result == "Contact person_01@company_01.com for details"


def test_apply_mapping_to_text_email_not_partial():
    mapping = {"john@acme.com": "person_01@company_01.com"}
    result = apply_mapping_to_text("andjohn@acme.comx stays", mapping)
    assert result == "andjohn@acme.comx stays"


def test_apply_mapping_to_text_phone_number():
    mapping = {"555-0101": "555-XXXX"}
    result = apply_mapping_to_text("Call 555-0101 today", mapping)
    assert result == "Call 555-XXXX today"


def test_apply_mapping_to_text_hyphenated_name():
    mapping = {"Mary-Jane": "PERSON_01"}
    result = apply_mapping_to_text("Mary-Jane Watson and Mary spoke", mapping)
    assert result == "PERSON_01 Watson and Mary spoke"


def test_apply_mapping_to_text_entity_at_string_boundaries():
    mapping = {"John": "PERSON_01"}
    assert apply_mapping_to_text("John", mapping) == "PERSON_01"
    assert apply_mapping_to_text("John.", mapping) == "PERSON_01."
    assert apply_mapping_to_text("(John)", mapping) == "(PERSON_01)"


def test_apply_mapping_to_text_multiple_occurrences():
    mapping = {"John": "PERSON_01"}
    result = apply_mapping_to_text("John met John at John's place", mapping)
    assert result == "PERSON_01 met PERSON_01 at PERSON_01's place"


def test_apply_mapping_to_text_adjacent_punctuation():
    mapping = {"Microsoft": "COMPANY_01"}
    result = apply_mapping_to_text('Microsoft, Microsoft. "Microsoft"', mapping)
    assert result == 'COMPANY_01, COMPANY_01. "COMPANY_01"'


def test_apply_mapping_to_text_case_sensitive():
    mapping = {"John": "PERSON_01"}
    result = apply_mapping_to_text("john is not John", mapping)
    assert result == "john is not PERSON_01"


def test_apply_mapping_preserving_text_structure_json_with_multiple_entities():
    original = json.dumps({"plaintiff": "John Smith", "defendant": "Microsoft Corp"})
    mapping = {"John Smith": "PERSON_01", "Microsoft Corp": "COMPANY_01"}
    result = apply_mapping_preserving_text_structure(original, mapping)
    parsed = json.loads(result)
    assert parsed == {"plaintiff": "PERSON_01", "defendant": "COMPANY_01"}


def test_apply_mapping_preserving_text_structure_json():
    original = json.dumps({"name": "John", "age": 30})
    result = apply_mapping_preserving_text_structure(original, {"John": "PERSON_01"})
    parsed = json.loads(result)
    assert parsed == {"name": "PERSON_01", "age": 30}
