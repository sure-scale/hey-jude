import json
import re
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

# Minimal, deliberately uninformative descriptor used to overwrite the
# descriptor of any entity the re-identification critic manages to pin down.
_GENERIC_DESCRIPTOR_BY_TYPE = {
    "PERSON": "a person",
    "ORGANIZATION": "an organization",
    "COMPANY": "a company",
    "LOCATION": "a location",
    "GPE": "a location",
    "EMAIL_ADDRESS": "an email address",
    "PHONE_NUMBER": "a phone number",
}

_GENERIC_PLACEHOLDER_TERMS = {
    "address",
    "agreement",
    "company",
    "consultant",
    "date",
    "director",
    "document",
    "employee",
    "employer",
    "executive",
    "person",
}


def _replacement_leaks_original(original: str, replacement: str) -> bool:
    if len(original) < 3:
        return False

    normalized_original = original.strip().casefold()
    if normalized_original in _GENERIC_PLACEHOLDER_TERMS:
        return False

    return normalized_original in replacement.casefold()


def validate_llm_anonymization_response(
    parsed: dict,
) -> tuple[list[FoundEntity], dict[str, str], str]:
    entities_raw = parsed.get("entities_found", [])
    descriptors = parsed.get("context_descriptors", {})
    sensitivity = parsed.get("sensitivity", "low")

    entities: list[FoundEntity] = []
    for item in entities_raw:
        action = item.get("action", "keep")
        replacement = item.get("replacement")

        if action in ("replace", "generalize") and not replacement:
            raise ValueError(
                f"Entity {item.get('text')!r} has action={action!r} but missing replacement"
            )

        if replacement:
            original = item.get("text", "")
            if _replacement_leaks_original(original, replacement):
                raise ValueError(
                    f"Replacement for {original!r} leaks original text: {replacement!r}"
                )
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
    """Substitute the per-request variables into the prompt template.

    The template deliberately keeps its static blocks (task, instructions,
    schema) first and both variables (`existing_mapping`, `message_text`) last.
    That ordering makes the large static portion a stable byte prefix so the LLM
    provider's automatic prompt caching can reuse it across every request.
    Keep the variables at the end when editing the template, or caching breaks.
    """
    mapping_str = json.dumps(existing_mapping, indent=2) if existing_mapping else "{}"
    return (
        template
        .replace("{message_text}", message_text)
        .replace("{existing_mapping}", mapping_str)
    )


def _parse_critic_guesses(raw: str) -> list[dict]:
    """Parse the critic's JSON, salvaging guesses from malformed output.

    The critic runs on the same local LLM as anonymization and can truncate or
    fence its JSON. Rather than discard the whole pass, fall back to harvesting
    individual guess objects so a re-identification is never missed on a parse
    error.
    """
    try:
        parsed = parse_llm_response(raw)
        guesses = parsed.get("guesses")
        if isinstance(guesses, list):
            return [g for g in guesses if isinstance(g, dict)]
    except ValueError:
        pass

    salvaged: list[dict] = []
    for match in re.finditer(r"\{[^{}]*\}", raw):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "placeholder" in obj and "guess" in obj:
            salvaged.append(obj)
    return salvaged


def _guess_matches_original(guess: str, original: str) -> bool:
    """Loose match between a critic guess and the true entity behind a placeholder."""
    g = guess.strip().casefold()
    o = original.strip().casefold()
    if not g or not o:
        return False
    if g == o:
        return True
    # Substring match, but only on meaningful spans to avoid e.g. "Inc" hitting
    # every company.
    if len(g) >= 4 and g in o:
        return True
    if len(o) >= 4 and o in g:
        return True
    g_tokens = {t for t in re.split(r"[^a-z0-9]+", g) if len(t) >= 4}
    o_tokens = {t for t in re.split(r"[^a-z0-9]+", o) if len(t) >= 4}
    return bool(g_tokens & o_tokens)


async def _run_reid_critic(
    sanitized_messages: list[ChatMessage],
    descriptors: dict[str, str],
    reverse_mapping: dict[str, str],
    placeholder_types: dict[str, str],
    settings: Settings,
) -> dict[str, str]:
    """Blind re-identification pass; broaden descriptors the critic cracks.

    Returns a possibly-modified copy of ``descriptors``. The critic sees only
    the sanitized text and descriptors (never the originals or the mapping);
    the pipeline scores its guesses against the truth it holds and overwrites
    any pinned-down descriptor with a generic, uninformative one.
    """
    sanitized_text = "\n\n".join(
        m.content for m in sanitized_messages if isinstance(m.content, str)
    )
    if not sanitized_text.strip():
        return descriptors

    template = Path(settings.reid_critic_prompt_path).read_text()
    descriptor_block = (
        "\n".join(f"- {name}: {desc}" for name, desc in descriptors.items())
        or "(none)"
    )
    prompt = (
        template
        .replace("{descriptors}", descriptor_block)
        .replace("{sanitized_text}", sanitized_text)
    )

    try:
        raw = await call_local_llm(prompt, settings)
    except Exception:
        # The critic is a best-effort hardening pass; never fail the request on it.
        return descriptors

    updated = dict(descriptors)
    for guess in _parse_critic_guesses(raw):
        placeholder = guess.get("placeholder")
        guess_text = guess.get("guess")
        try:
            confidence = float(guess.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not placeholder or not guess_text:
            continue
        if confidence < settings.reid_critic_threshold:
            continue
        # Only broaden a descriptor that already exists. If the critic cracked a
        # placeholder that had no descriptor, the leak came from the text or
        # structure — inventing a descriptor here would add signal, not remove it.
        if placeholder not in updated:
            continue
        original = reverse_mapping.get(placeholder)
        if not original or not _guess_matches_original(guess_text, original):
            continue
        entity_type = placeholder_types.get(placeholder, "").upper()
        updated[placeholder] = _GENERIC_DESCRIPTOR_BY_TYPE.get(
            entity_type, "an entity referenced in the document"
        )
    return updated


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

    if settings.reid_critic_enabled and overall_sensitivity == "high":
        placeholder_types = {
            e.replacement: e.entity_type
            for e in all_entities
            if e.replacement
        }
        sanitized_descriptors = await _run_reid_critic(
            sanitized_messages,
            sanitized_descriptors,
            reverse_mapping,
            placeholder_types,
            settings,
        )

    return AnonymizationResult(
        mapping=accumulated_mapping,
        reverse_mapping=reverse_mapping,
        context_descriptors=sanitized_descriptors,
        sanitized_messages=sanitized_messages,
        sensitivity=overall_sensitivity,
        entities_found=all_entities,
    )
