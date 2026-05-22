import json
from pathlib import Path

from hey_jude.config import Settings
from hey_jude.models import (
    AnonymizationResult,
    ChatMessage,
    FoundEntity,
)
from hey_jude.services.llm_client import call_local_llm, parse_llm_response
from hey_jude.services.text import (
    apply_mapping_to_text,
    apply_mapping_preserving_text_structure,
)


def validate_llm_anonymization_response(
    parsed: dict,
) -> tuple[list[FoundEntity], dict[str, str], str]:
    entities_raw = parsed.get("entities_found", [])
    descriptors = parsed.get("context_descriptors", {})
    sensitivity = parsed.get("sensitivity", "low")

    entities: list[FoundEntity] = []
    replacements_seen: set[str] = set()

    for item in entities_raw:
        action = item.get("action", "keep")
        replacement = item.get("replacement")

        if action in ("replace", "generalize") and not replacement:
            raise ValueError(
                f"Entity {item.get('text')!r} has action={action!r} but missing replacement"
            )

        if replacement:
            original = item.get("text", "")
            if len(original) >= 3 and original.casefold() in replacement.casefold():
                raise ValueError(
                    f"Replacement for {original!r} leaks original text: {replacement!r}"
                )
            if replacement in replacements_seen:
                raise ValueError(
                    f"Found duplicate replacement {replacement!r} for {original!r}"
                )
            replacements_seen.add(replacement)

        entities.append(FoundEntity(
            text=item.get("text", ""),
            entity_type=item.get("type", "UNKNOWN"),
            action=action,
            replacement=replacement,
            reason=item.get("reason"),
        ))

    return entities, descriptors, sensitivity


def build_mapping_from_entities(entities: list[FoundEntity]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entity in entities:
        if entity.action in ("replace", "generalize") and entity.replacement:
            mapping[entity.text] = entity.replacement
    return mapping


def render_prompt(
    template: str,
    message_text: str,
    existing_mapping: dict[str, str],
) -> str:
    mapping_str = json.dumps(existing_mapping, indent=2) if existing_mapping else "{}"
    return (
        template
        .replace("{message_text}", message_text)
        .replace("{existing_mapping}", mapping_str)
    )


async def anonymize_messages(
    messages: list[ChatMessage],
    settings: Settings,
    prompt_template: str | None = None,
    existing_mapping: dict[str, str] | None = None,
) -> AnonymizationResult:
    if prompt_template is None:
        prompt_template = Path(settings.anonymization_prompt_path).read_text()

    accumulated_mapping: dict[str, str] = dict(existing_mapping or {})
    all_entities: list[FoundEntity] = []
    all_descriptors: dict[str, str] = {}
    overall_sensitivity = "low"
    skip_indices: set[int] = set()

    for i, msg in enumerate(messages):
        if msg.role == "system" and settings.passthrough_system_messages:
            skip_indices.add(i)
            continue

        rendered = render_prompt(prompt_template, msg.content, accumulated_mapping)

        last_error: ValueError | None = None
        for attempt in range(2):
            prompt = rendered if attempt == 0 else (
                rendered
                + "\n\nIMPORTANT: Your previous response was invalid. "
                + "Return ONLY a valid JSON object matching the requested schema. "
                + f"Validation error: {last_error}"
            )
            raw = await call_local_llm(prompt, settings)
            try:
                parsed = parse_llm_response(raw)
                entities, descriptors, sensitivity = validate_llm_anonymization_response(parsed)
                break
            except ValueError as e:
                last_error = e
        else:
            raise last_error or ValueError("LLM returned invalid anonymization response")

        new_mapping = build_mapping_from_entities(entities)
        for key, value in new_mapping.items():
            accumulated_mapping.setdefault(key, value)
        all_entities.extend(entities)
        all_descriptors.update(descriptors)
        if sensitivity == "high":
            overall_sensitivity = "high"

    sanitized_messages = []
    for i, msg in enumerate(messages):
        if i in skip_indices:
            sanitized_messages.append(msg)
        else:
            sanitized_content = apply_mapping_preserving_text_structure(
                msg.content, accumulated_mapping
            )
            sanitized_messages.append(ChatMessage(role=msg.role, content=sanitized_content))

    sanitized_descriptors = {
        apply_mapping_to_text(k, accumulated_mapping): apply_mapping_to_text(v, accumulated_mapping)
        for k, v in all_descriptors.items()
    }

    reverse_mapping = {v: k for k, v in accumulated_mapping.items()}

    return AnonymizationResult(
        mapping=accumulated_mapping,
        reverse_mapping=reverse_mapping,
        context_descriptors=sanitized_descriptors,
        sanitized_messages=sanitized_messages,
        sensitivity=overall_sensitivity,
        entities_found=all_entities,
    )
