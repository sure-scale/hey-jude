import json
import re


def apply_mapping_to_text(text: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return text
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    parts = []
    for key in sorted_keys:
        escaped = re.escape(key)
        if re.match(r"\w", key[0]):
            escaped = r"\b" + escaped
        if re.match(r"\w", key[-1]):
            escaped = escaped + r"\b"
        parts.append(escaped)
    pattern = re.compile("|".join(parts))
    return pattern.sub(lambda m: mapping[m.group(0)], text)


def replace_json_string_values(value: object, mapping: dict[str, str]) -> object:
    if isinstance(value, str):
        return apply_mapping_to_text(value, mapping)
    if isinstance(value, list):
        return [replace_json_string_values(item, mapping) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_json_string_values(item, mapping)
            for key, item in value.items()
        }
    return value


def apply_mapping_preserving_text_structure(text: str, mapping: dict[str, str]) -> str:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return apply_mapping_to_text(text, mapping)
    if not isinstance(parsed, (dict, list)):
        return apply_mapping_to_text(text, mapping)
    replaced = replace_json_string_values(parsed, mapping)
    return json.dumps(replaced, separators=(",", ":"))
