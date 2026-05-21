# Placeholder Substitution Strategy

**Issue:** [#6 — Add structured placeholder substitution strategy](https://github.com/sure-scale/hey-jude/issues/6)  
**Date:** 2026-05-21  
**Status:** Approved

## Summary

Add a third substitution strategy (`"placeholder"`) to hey-jude's PII anonymization pipeline. Entities configured with this strategy are replaced with structured tokens like `PERSON_01`, `COMPANY_01` — no local LLM call required.

Evaluation results (168 tests, 28 legal scenarios, 2 models) show placeholder wins 93% head-to-head vs fictional names on de-anonymized output quality. The decisive factor: LLMs abbreviate fictional names ("Sterling Capital Partners" → "Sterling"), which survives reverse-mapping. Placeholder tokens are never abbreviated.

## Approach

Minimal strategy function — same pattern as existing `_build_deterministic_replacements()`. No architectural changes, no strategy registry, no class hierarchy. Three similar functions is the right call at three strategies.

## Design

### New function: `_build_placeholder_replacements()`

Location: `src/hey_jude/services/substitutor.py`

```python
_PLACEHOLDER_LABELS = {
    "ORGANIZATION": "COMPANY",
}

def _build_placeholder_replacements(
    entities: list[DetectedEntity],
    strategies: dict[str, str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}
    for entity in entities:
        if strategies.get(entity.entity_type) != "placeholder":
            continue
        if entity.text in mapping:
            continue
        label = _PLACEHOLDER_LABELS.get(entity.entity_type, entity.entity_type)
        counters[label] = counters.get(label, 0) + 1
        mapping[entity.text] = f"{label}_{counters[label]:02d}"
    return mapping
```

- Filters entities on `"placeholder"` strategy
- Maps `ORGANIZATION` → `COMPANY` for LLM readability (matches evaluation convention)
- All other entity types use Presidio type name directly (`PERSON_01`, `EMAIL_ADDRESS_01`)
- Zero-padded two-digit counter
- Deduplicates: same entity text → same placeholder

### Wiring in `substitute_entities()`

New phase order: **placeholder → deterministic → LLM → merge**.

```python
placeholder_mapping = _build_placeholder_replacements(
    entities, settings.entity_strategies
)
# ... existing deterministic_mapping and llm_mapping ...
full_mapping = {**placeholder_mapping, **deterministic_mapping, **llm_mapping}
```

No changes to LLM call gating — when no entities have `"llm"` strategy, `should_call_local` remains `False` and the LLM call is skipped automatically.

### Config default change

Location: `src/hey_jude/config.py`

```python
entity_strategies: dict[str, str] = Field(
    default={
        "PERSON": "placeholder",
        "ORGANIZATION": "placeholder",
        "EMAIL_ADDRESS": "deterministic",
        "PHONE_NUMBER": "deterministic",
    }
)
```

PERSON and ORGANIZATION switch from `"llm"` to `"placeholder"`. EMAIL_ADDRESS and PHONE_NUMBER remain `"deterministic"`. Users can configure any entity type to use any strategy via config.

### Metadata behavior (no LLM phase)

When no entities require the `"llm"` strategy, the substitution result uses sensible defaults:

- `sensitivity`: `"low"`
- `context_descriptors`: `{}`
- `needs_clarification`: `False`
- `clarification_question`: `None`

These are already the defaults in `substitute_entities()` — no code change needed.

### Tests

New tests in `tests/test_substitutor.py`:

1. `test_build_placeholder_replacements` — format, counter, dedup
2. `test_build_placeholder_skips_non_placeholder_entities` — only processes placeholder-strategy entities
3. `test_substitute_entities_placeholder_only_skips_llm_call` — no LLM mock, no LLM call
4. `test_substitute_entities_mixed_placeholder_and_deterministic` — both strategies coexist
5. `test_substitute_entities_mixed_placeholder_and_llm` — placeholder + LLM, both produce correct replacements

### Files changed

| File | Change |
|------|--------|
| `src/hey_jude/services/substitutor.py` | Add `_PLACEHOLDER_LABELS`, `_build_placeholder_replacements()`, wire into `substitute_entities()` |
| `src/hey_jude/config.py` | Update default `entity_strategies` (PERSON/ORGANIZATION → placeholder) |
| `tests/test_substitutor.py` | Add 5 new test functions |

### Files NOT changed

- `models.py` — `SubstitutionResult` schema unchanged
- `services/mapper.py` — reverse mapping works by exact match, handles placeholder tokens identically
- `routes.py` — no API changes
- `services/detector.py` — entity detection unchanged

## Out of scope

- **Sub-typed placeholders** (e.g., `TECH_COMPANY_01`): requires LLM classification. Tracked as separate issue.
- **LLM-based entity detection**: replacing Presidio with LLM classification. Future work.
- **Strategy validation**: config doesn't validate strategy names. Unrecognized strategies silently produce no replacement. Acceptable for now.
