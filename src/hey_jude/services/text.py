import json


def apply_mapping_to_text(text: str, mapping: dict[str, str]) -> str:
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    for original in sorted_keys:
        replacement = mapping[original]
        text = text.replace(original, replacement)
    return text


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
