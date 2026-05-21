# Placeholder Substitution Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `"placeholder"` substitution strategy that replaces PII entities with structured tokens (`PERSON_01`, `COMPANY_01`) — no LLM call needed.

**Architecture:** New `_build_placeholder_replacements()` function in `substitutor.py`, same pattern as existing `_build_deterministic_replacements()`. Wired into `substitute_entities()` as first phase before deterministic and LLM. Config defaults updated so PERSON/ORGANIZATION use placeholder instead of LLM.

**Tech Stack:** Python, pytest, pytest-asyncio, Pydantic

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/hey_jude/services/substitutor.py` | Modify | Add `_PLACEHOLDER_LABELS` dict, `_build_placeholder_replacements()` function, wire into `substitute_entities()` |
| `src/hey_jude/config.py` | Modify | Change default `entity_strategies` for PERSON/ORGANIZATION from `"llm"` to `"placeholder"` |
| `tests/test_substitutor.py` | Modify | Add 5 new test functions, import `_build_placeholder_replacements` |

**Note on test isolation:** `tests/conftest.py` sets `ENTITY_STRATEGIES` env var with LLM defaults for PERSON/ORGANIZATION. This means existing tests using `Settings()` keep working after the config.py default change. New placeholder tests pass explicit `Settings(entity_strategies=...)` to control strategy selection.

---

### Task 1: `_build_placeholder_replacements()` — Unit Tests + Implementation

**Files:**
- Modify: `tests/test_substitutor.py:1-11` (imports), then append tests after line 34
- Modify: `src/hey_jude/services/substitutor.py:25` (insert before `_build_deterministic_replacements`)

- [ ] **Step 1: Add import for new function in test file**

At line 5 of `tests/test_substitutor.py`, add `_build_placeholder_replacements` to the import:

```python
from hey_jude.services.substitutor import (
    _build_deterministic_replacements,
    _build_placeholder_replacements,
    _build_prompt,
    _call_local_llm,
    _parse_llm_response,
    substitute_entities,
)
```

- [ ] **Step 2: Write failing test for placeholder format, counter, and dedup**

Add after `test_build_deterministic_skips_llm_entities` (after line 34):

```python
def test_build_placeholder_replacements():
    entities = [
        DetectedEntity(text="John Smith", entity_type="PERSON", start=0, end=10, score=0.9),
        DetectedEntity(text="Jane Doe", entity_type="PERSON", start=15, end=23, score=0.9),
        DetectedEntity(text="Acme Corp", entity_type="ORGANIZATION", start=30, end=39, score=0.9),
        DetectedEntity(text="John Smith", entity_type="PERSON", start=45, end=55, score=0.9),
    ]
    strategies = {"PERSON": "placeholder", "ORGANIZATION": "placeholder"}
    mapping = _build_placeholder_replacements(entities, strategies)
    assert mapping["John Smith"] == "PERSON_01"
    assert mapping["Jane Doe"] == "PERSON_02"
    assert mapping["Acme Corp"] == "COMPANY_01"
    assert len(mapping) == 3


def test_build_placeholder_skips_non_placeholder_entities():
    entities = [
        DetectedEntity(text="John Smith", entity_type="PERSON", start=0, end=10, score=0.9),
        DetectedEntity(text="john@acme.com", entity_type="EMAIL_ADDRESS", start=15, end=28, score=0.9),
        DetectedEntity(text="Acme Corp", entity_type="ORGANIZATION", start=35, end=44, score=0.9),
    ]
    strategies = {
        "PERSON": "placeholder",
        "ORGANIZATION": "placeholder",
        "EMAIL_ADDRESS": "deterministic",
    }
    mapping = _build_placeholder_replacements(entities, strategies)
    assert "John Smith" in mapping
    assert "Acme Corp" in mapping
    assert "john@acme.com" not in mapping
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_substitutor.py::test_build_placeholder_replacements tests/test_substitutor.py::test_build_placeholder_skips_non_placeholder_entities -v`

Expected: `ImportError` — `_build_placeholder_replacements` does not exist yet.

- [ ] **Step 4: Implement `_build_placeholder_replacements()` and `_PLACEHOLDER_LABELS`**

Insert before `_build_deterministic_replacements` (before line 25 of `substitutor.py`):

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

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_substitutor.py::test_build_placeholder_replacements tests/test_substitutor.py::test_build_placeholder_skips_non_placeholder_entities -v`

Expected: Both PASS.

- [ ] **Step 6: Run full existing test suite to verify no regressions**

Run: `python -m pytest tests/test_substitutor.py -v`

Expected: All existing tests still pass (conftest env var overrides config defaults).

- [ ] **Step 7: Commit**

```bash
git add src/hey_jude/services/substitutor.py tests/test_substitutor.py
git commit -m "feat: add _build_placeholder_replacements() with unit tests

Implements the placeholder mapping function that replaces entities with
structured tokens (PERSON_01, COMPANY_01). Maps ORGANIZATION -> COMPANY
for LLM readability. Not yet wired into substitute_entities().

Refs #6"
```

---

### Task 2: Wire Placeholder Into `substitute_entities()` + Integration Tests

**Files:**
- Modify: `src/hey_jude/services/substitutor.py:302-360` (`substitute_entities` function)
- Modify: `tests/test_substitutor.py` (append 3 new async tests)

- [ ] **Step 1: Write failing test — placeholder-only skips LLM call**

Append to `tests/test_substitutor.py`:

```python
async def test_substitute_entities_placeholder_only_skips_llm_call():
    messages = [
        ChatMessage(role="user", content="John Smith works at Acme Corp")
    ]
    entities = [
        DetectedEntity(text="John Smith", entity_type="PERSON", start=0, end=10, score=0.9),
        DetectedEntity(text="Acme Corp", entity_type="ORGANIZATION", start=20, end=29, score=0.9),
    ]
    placeholder_settings = Settings(
        entity_strategies={
            "PERSON": "placeholder",
            "ORGANIZATION": "placeholder",
        }
    )

    with patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local_llm:
        result = await substitute_entities(messages, entities, placeholder_settings)

    mock_local_llm.assert_not_awaited()
    assert result.mapping == {"John Smith": "PERSON_01", "Acme Corp": "COMPANY_01"}
    assert result.sanitized_messages[0].content == "PERSON_01 works at COMPANY_01"
    assert result.sensitivity == "low"
    assert result.context_descriptors == {}
    assert result.needs_clarification is False
```

- [ ] **Step 2: Write failing test — mixed placeholder and deterministic**

Append to `tests/test_substitutor.py`:

```python
async def test_substitute_entities_mixed_placeholder_and_deterministic():
    messages = [
        ChatMessage(role="user", content="John Smith's email is john@acme.com")
    ]
    entities = [
        DetectedEntity(text="John Smith", entity_type="PERSON", start=0, end=10, score=0.9),
        DetectedEntity(text="john@acme.com", entity_type="EMAIL_ADDRESS", start=21, end=34, score=0.9),
    ]
    mixed_settings = Settings(
        entity_strategies={
            "PERSON": "placeholder",
            "EMAIL_ADDRESS": "deterministic",
        }
    )

    with patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local_llm:
        result = await substitute_entities(messages, entities, mixed_settings)

    mock_local_llm.assert_not_awaited()
    assert result.mapping["John Smith"] == "PERSON_01"
    assert result.mapping["john@acme.com"] == "user_1@example.com"
    assert result.sanitized_messages[0].content == (
        "PERSON_01's email is user_1@example.com"
    )
```

- [ ] **Step 3: Write failing test — mixed placeholder and LLM**

Append to `tests/test_substitutor.py`:

```python
async def test_substitute_entities_mixed_placeholder_and_llm():
    raw = json.dumps({
        "sensitivity": "high",
        "reasoning": "Named relationship could identify the person.",
        "mapping": {"Acme Corp": "TechNova Inc."},
        "context_descriptors": {"TechNova Inc.": "technology company"},
        "sanitized_text": "",
        "needs_clarification": False,
        "clarification_question": None,
    })
    messages = [
        ChatMessage(role="user", content="John Smith works at Acme Corp")
    ]
    entities = [
        DetectedEntity(text="John Smith", entity_type="PERSON", start=0, end=10, score=0.9),
        DetectedEntity(text="Acme Corp", entity_type="ORGANIZATION", start=20, end=29, score=0.9),
    ]
    mixed_settings = Settings(
        entity_strategies={
            "PERSON": "placeholder",
            "ORGANIZATION": "llm",
        }
    )

    with patch(
        "hey_jude.services.substitutor._call_local_llm",
        new_callable=AsyncMock,
    ) as mock_local_llm:
        mock_local_llm.return_value = raw
        result = await substitute_entities(messages, entities, mixed_settings)

    mock_local_llm.assert_awaited_once()
    assert result.mapping["John Smith"] == "PERSON_01"
    assert result.mapping["Acme Corp"] == "TechNova Inc."
    assert result.sanitized_messages[0].content == (
        "PERSON_01 works at TechNova Inc."
    )
    assert result.sensitivity == "high"
```

- [ ] **Step 4: Run new tests to verify they fail**

Run: `python -m pytest tests/test_substitutor.py::test_substitute_entities_placeholder_only_skips_llm_call tests/test_substitutor.py::test_substitute_entities_mixed_placeholder_and_deterministic tests/test_substitutor.py::test_substitute_entities_mixed_placeholder_and_llm -v`

Expected: FAIL — placeholder entities produce no mapping (strategy `"placeholder"` not handled yet in `substitute_entities`).

- [ ] **Step 5: Wire placeholder phase into `substitute_entities()`**

In `src/hey_jude/services/substitutor.py`, modify `substitute_entities()`:

Before the existing `deterministic_mapping` line (line 308), add the placeholder phase:

```python
    placeholder_mapping = _build_placeholder_replacements(
        entities, settings.entity_strategies
    )
```

Then change the merge line (line 360) from:

```python
    full_mapping = {**deterministic_mapping, **llm_mapping}
```

to:

```python
    full_mapping = {**placeholder_mapping, **deterministic_mapping, **llm_mapping}
```

- [ ] **Step 6: Run new tests to verify they pass**

Run: `python -m pytest tests/test_substitutor.py::test_substitute_entities_placeholder_only_skips_llm_call tests/test_substitutor.py::test_substitute_entities_mixed_placeholder_and_deterministic tests/test_substitutor.py::test_substitute_entities_mixed_placeholder_and_llm -v`

Expected: All 3 PASS.

- [ ] **Step 7: Run full test suite for regressions**

Run: `python -m pytest tests/test_substitutor.py -v`

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/hey_jude/services/substitutor.py tests/test_substitutor.py
git commit -m "feat: wire placeholder strategy into substitute_entities()

Placeholder phase runs before deterministic and LLM phases. When no
entities need LLM strategy, the LLM call is skipped automatically.
Adds integration tests for placeholder-only, placeholder+deterministic,
and placeholder+LLM combinations.

Refs #6"
```

---

### Task 3: Update Config Defaults + Create GitHub Issue for Future Work

**Files:**
- Modify: `src/hey_jude/config.py:28-35`
- Modify: `tests/conftest.py:11-14` (update env var default to match new config)

- [ ] **Step 1: Update config.py default entity_strategies**

In `src/hey_jude/config.py`, change lines 28-35 from:

```python
    entity_strategies: dict[str, str] = Field(
        default={
            "PERSON": "llm",
            "ORGANIZATION": "llm",
            "EMAIL_ADDRESS": "deterministic",
            "PHONE_NUMBER": "deterministic",
        }
    )
```

to:

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

- [ ] **Step 2: Update conftest.py env var to match new defaults**

In `tests/conftest.py`, change lines 11-14 from:

```python
os.environ.setdefault(
    "ENTITY_STRATEGIES",
    '{"PERSON":"llm","ORGANIZATION":"llm","EMAIL_ADDRESS":"deterministic","PHONE_NUMBER":"deterministic"}',
)
```

to:

```python
os.environ.setdefault(
    "ENTITY_STRATEGIES",
    '{"PERSON":"placeholder","ORGANIZATION":"placeholder","EMAIL_ADDRESS":"deterministic","PHONE_NUMBER":"deterministic"}',
)
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/test_substitutor.py -v`

Expected: All tests pass. Existing LLM-strategy tests pass because they supply explicit `Settings()` which picks up conftest env var, and tests that mock `_call_local_llm` use `Settings()` with the default strategies — but those tests also supply entities with types that match whatever strategy is active. Let me verify:

- Tests at lines 102-233 use `Settings()` and mock `_call_local_llm`. These tests have entities with types ORGANIZATION/PERSON which now default to `"placeholder"`. The mock would NOT be called. But these tests expect LLM behavior (mappings like "Vertex Holdings", "Pinnacle Systems").

**This means existing tests need explicit LLM strategies.** Each existing `substitute_entities` test that mocks the LLM must pass `Settings(entity_strategies={"PERSON": "llm", "ORGANIZATION": "llm", ...})`.

Update each existing async test that calls `substitute_entities` with `Settings()` to use explicit LLM strategies:

For tests at lines 102, 145, 194, 235, 282, 330, 359 — replace `Settings()` with:

```python
Settings(
    entity_strategies={
        "PERSON": "llm",
        "ORGANIZATION": "llm",
        "EMAIL_ADDRESS": "deterministic",
        "PHONE_NUMBER": "deterministic",
    }
)
```

The affected calls are:
- Line 138: `await substitute_entities(messages, entities, Settings())`
- Line 183: `await substitute_entities(messages, entities, Settings())`
- Line 224: `await substitute_entities(messages, entities, Settings())`
- Line 271: `await substitute_entities(messages, entities, Settings())`
- Line 318: `await substitute_entities(messages, entities, Settings())`
- Line 354: `await substitute_entities(messages, entities, Settings())`
- Line 396: `await substitute_entities(messages, entities, Settings())`

Each becomes `await substitute_entities(messages, entities, Settings(entity_strategies={"PERSON": "llm", "ORGANIZATION": "llm", "EMAIL_ADDRESS": "deterministic", "PHONE_NUMBER": "deterministic"}))`.

Alternatively, add a helper at the top of the test file:

```python
_LLM_STRATEGIES = {
    "PERSON": "llm",
    "ORGANIZATION": "llm",
    "EMAIL_ADDRESS": "deterministic",
    "PHONE_NUMBER": "deterministic",
}
```

Then each call becomes `Settings(entity_strategies=_LLM_STRATEGIES)`.

- [ ] **Step 4: Run full test suite to confirm all pass**

Run: `python -m pytest tests/test_substitutor.py -v`

Expected: All tests pass — existing LLM tests use explicit LLM strategies, new placeholder tests use explicit placeholder strategies.

- [ ] **Step 5: Create GitHub issue for sub-typed placeholders**

Run:
```bash
gh issue create --repo sure-scale/hey-jude \
  --title "Add semantic sub-typed placeholders (TECH_COMPANY_01, LAW_FIRM_01)" \
  --body "$(cat <<'EOF'
## Summary

Extend the placeholder substitution strategy to support semantic sub-types. Instead of generic `COMPANY_01`, produce `TECH_COMPANY_01`, `LAW_FIRM_01`, etc. based on entity context.

## Motivation

During brainstorming for #6, the idea came up that sub-typed placeholders could give LLMs richer context without the de-anonymization risks of fictional names. For example, knowing an entity is a "tech company" vs a "law firm" helps the LLM reason about industry-specific regulations.

## Requirements

- Classification of entities into sub-types (requires LLM or taxonomy)
- Mapping from sub-type to placeholder label
- Must not increase information leakage risk (sub-types should be broad enough to not narrow identification)

## Depends on

- #6 (placeholder strategy must land first)

## Open questions

- LLM-based classification vs static taxonomy?
- How granular should sub-types be? (too specific = leakage risk)
- Should this be a separate strategy or an option on the existing placeholder strategy?
EOF
)"
```

- [ ] **Step 6: Commit**

```bash
git add src/hey_jude/config.py tests/conftest.py tests/test_substitutor.py
git commit -m "feat: set placeholder as default strategy for PERSON/ORGANIZATION

Updates config defaults and test fixtures. Existing LLM-strategy tests
now use explicit entity_strategies to avoid depending on defaults.

Refs #6"
```

---

### Task 4: Final Verification

- [ ] **Step 1: Run complete test suite**

Run: `python -m pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 2: Verify no untracked or unstaged changes**

Run: `git status`

Expected: Clean working tree (except `tests/test_substitution_quality.py` which was pre-existing untracked).
