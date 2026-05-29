from dataclasses import dataclass

from presidio_analyzer import Pattern, PatternRecognizer

from hey_jude.services.config_files import load_structured_file

_DEFAULT_STRATEGY = "placeholder"


@dataclass(frozen=True)
class PatternSpec:
    regex: str
    score: float


@dataclass(frozen=True)
class RecognizerSpec:
    name: str
    entity_type: str
    patterns: tuple[PatternSpec, ...]
    context: tuple[str, ...] = ()
    strategy: str = _DEFAULT_STRATEGY


def _parse_pattern(raw: object, recognizer_name: str) -> PatternSpec:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Custom recognizer {recognizer_name!r}: each pattern must be a mapping "
            f"with 'regex' and 'score', got {type(raw).__name__}"
        )
    regex = raw.get("regex")
    if not isinstance(regex, str) or not regex:
        raise ValueError(
            f"Custom recognizer {recognizer_name!r}: pattern is missing a non-empty 'regex'"
        )
    score = raw.get("score", 0.85)
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ValueError(
            f"Custom recognizer {recognizer_name!r}: pattern 'score' must be a number"
        )
    if not 0.0 <= float(score) <= 1.0:
        raise ValueError(
            f"Custom recognizer {recognizer_name!r}: pattern 'score' must be between 0 and 1"
        )
    return PatternSpec(regex=regex, score=float(score))


def _parse_recognizer(raw: object) -> RecognizerSpec:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Each custom recognizer must be a mapping, got {type(raw).__name__}"
        )
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Each custom recognizer needs a non-empty 'name'")
    entity_type = raw.get("entity_type")
    if not isinstance(entity_type, str) or not entity_type:
        raise ValueError(f"Custom recognizer {name!r}: needs a non-empty 'entity_type'")

    patterns_raw = raw.get("patterns")
    if not isinstance(patterns_raw, list) or not patterns_raw:
        raise ValueError(
            f"Custom recognizer {name!r}: needs a non-empty 'patterns' list"
        )
    patterns = tuple(_parse_pattern(p, name) for p in patterns_raw)

    context_raw = raw.get("context", [])
    if not isinstance(context_raw, list) or not all(
        isinstance(c, str) for c in context_raw
    ):
        raise ValueError(
            f"Custom recognizer {name!r}: 'context' must be a list of strings"
        )

    strategy = raw.get("strategy", _DEFAULT_STRATEGY)
    if not isinstance(strategy, str) or not strategy:
        raise ValueError(
            f"Custom recognizer {name!r}: 'strategy' must be a non-empty string"
        )

    return RecognizerSpec(
        name=name,
        entity_type=entity_type,
        patterns=patterns,
        context=tuple(context_raw),
        strategy=strategy,
    )


def load_recognizer_specs(path: str) -> list[RecognizerSpec]:
    data = load_structured_file(path, "Custom recognizers")
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(
            f"Custom recognizers config ({path}) must be a list of recognizers, "
            f"got {type(data).__name__}"
        )
    return [_parse_recognizer(item) for item in data]


def build_recognizers(specs: list[RecognizerSpec]) -> list[PatternRecognizer]:
    recognizers = []
    for spec in specs:
        patterns = [
            Pattern(name=spec.name, regex=p.regex, score=p.score)
            for p in spec.patterns
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity=spec.entity_type,
                name=spec.name,
                patterns=patterns,
                context=list(spec.context) or None,
            )
        )
    return recognizers
