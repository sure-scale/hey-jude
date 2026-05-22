import logging
import re

from hey_jude.config import Settings
from hey_jude.models import ChatMessage, DetectedEntity, SafetyNetResult
from hey_jude.services.detector import detect_entities
from hey_jude.services.text import apply_mapping_preserving_text_structure

logger = logging.getLogger(__name__)

_PLACEHOLDER_PATTERN = re.compile(r"^[A-Z][A-Z_]+_\d+$")


def _is_own_placeholder(text: str, mapping_values: set[str]) -> bool:
    if text in mapping_values:
        return True
    if _PLACEHOLDER_PATTERN.match(text):
        return True
    for value in mapping_values:
        if text in value and _PLACEHOLDER_PATTERN.match(value):
            return True
    return False


async def safety_net_check(
    sanitized_messages: list[ChatMessage],
    mapping: dict[str, str],
    settings: Settings,
) -> SafetyNetResult:
    if settings.safety_net_strictness == "off":
        return SafetyNetResult(passed=True, leaked_entities=[], auto_replaced=0)

    mapping_values = set(mapping.values())
    leaked: list[DetectedEntity] = []

    for msg in sanitized_messages:
        detected = await detect_entities(msg.content, settings)
        for entity in detected:
            if _is_own_placeholder(entity.text, mapping_values):
                continue
            leaked.append(entity)

    if not leaked:
        return SafetyNetResult(passed=True, leaked_entities=[], auto_replaced=0)

    if settings.safety_net_strictness == "strict":
        return SafetyNetResult(passed=False, leaked_entities=leaked, auto_replaced=0)

    counters: dict[str, int] = {}
    for entity in leaked:
        label = entity.entity_type
        counters[label] = counters.get(label, 0) + 1
        placeholder = f"MISSED_{label}_{counters[label]:02d}"
        mapping[entity.text] = placeholder
        logger.warning(
            "Safety net caught leaked PII: %r (%s) → %s",
            entity.text, entity.entity_type, placeholder,
        )

    for i, msg in enumerate(sanitized_messages):
        sanitized_messages[i] = ChatMessage(
            role=msg.role,
            content=apply_mapping_preserving_text_structure(msg.content, mapping),
        )

    return SafetyNetResult(passed=True, leaked_entities=leaked, auto_replaced=len(leaked))
