import json
import re
from dataclasses import dataclass

from hey_jude.models import ChatMessage
from hey_jude.services.config_files import load_structured_file


@dataclass(frozen=True)
class KnownEntity:
    """A firm-maintained known entity and its spelling variants.

    `term` is the canonical form, `aliases` are alternate spellings/abbreviations.
    All variants map to a single placeholder so one client never produces two
    different aliases within a request. `replace_with`, if set, fixes the
    placeholder so it stays stable across requests too.
    """

    entity_type: str
    term: str
    aliases: tuple[str, ...] = ()
    replace_with: str | None = None


def _parse_known_entity(raw: object) -> KnownEntity:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Each known entity must be a mapping, got {type(raw).__name__}"
        )
    entity_type = raw.get("entity_type")
    if not isinstance(entity_type, str) or not entity_type:
        raise ValueError("Each known entity needs a non-empty 'entity_type'")
    term = raw.get("term")
    if not isinstance(term, str) or not term.strip():
        raise ValueError(
            f"Known entity of type {entity_type!r}: needs a non-empty 'term'"
        )

    aliases_raw = raw.get("aliases", [])
    if not isinstance(aliases_raw, list) or not all(
        isinstance(a, str) and a.strip() for a in aliases_raw
    ):
        raise ValueError(
            f"Known entity {term!r}: 'aliases' must be a list of non-empty strings"
        )

    replace_with = raw.get("replace_with")
    if replace_with is not None and (
        not isinstance(replace_with, str) or not replace_with.strip()
    ):
        raise ValueError(
            f"Known entity {term!r}: 'replace_with' must be a non-empty string when set"
        )

    return KnownEntity(
        entity_type=entity_type,
        term=term,
        aliases=tuple(aliases_raw),
        replace_with=replace_with,
    )


def load_known_entities(path: str) -> list[KnownEntity]:
    data = load_structured_file(path, "Known entities")
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(
            f"Known entities config ({path}) must be a list of entities, "
            f"got {type(data).__name__}"
        )
    return [_parse_known_entity(item) for item in data]


def _variant_pattern(variant: str) -> re.Pattern:
    escaped = re.escape(variant)
    if re.match(r"\w", variant[0]):
        escaped = r"\b" + escaped
    if re.match(r"\w", variant[-1]):
        escaped = escaped + r"\b"
    return re.compile(escaped, re.IGNORECASE)


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def seed_mapping(
    messages: list[ChatMessage],
    known: list[KnownEntity],
    settings,
) -> dict[str, str]:
    """Build a deterministic original->placeholder mapping for known entities.

    Matching is case-insensitive and word-boundary aware, but the captured
    surface form (real casing) is used as the mapping key so the replacement is
    guaranteed to apply regardless of how the entity was written in the prompt.
    """
    if not known:
        return {}

    texts = []
    for msg in messages:
        if msg.role == "system" and settings.passthrough_system_messages:
            continue
        texts.append(_message_text(msg.content))
    combined = "\n".join(texts)

    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}
    for entity in known:
        variants = sorted({entity.term, *entity.aliases}, key=len, reverse=True)
        surfaces = [
            match.group(0)
            for variant in variants
            for match in _variant_pattern(variant).finditer(combined)
        ]
        if not surfaces:
            continue

        if entity.replace_with:
            placeholder = entity.replace_with
        else:
            counters[entity.entity_type] = counters.get(entity.entity_type, 0) + 1
            placeholder = f"{entity.entity_type}_{counters[entity.entity_type]:02d}"

        for surface in surfaces:
            mapping[surface] = placeholder

    return mapping
