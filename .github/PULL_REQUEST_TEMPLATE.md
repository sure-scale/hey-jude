## Summary

<!-- What does this PR do and why? -->

Closes #

## How it was tested

```
uv run pytest
uv run pytest -m e2e   # if the change touches runtime behavior
```

<!-- Paste the pass/fail summary line. -->

## Checklist

- [ ] Tests pass (`uv run pytest`); `e2e` confirmed passing or N/A
- [ ] No real PII, secrets, or keys in the diff, tests, or fixtures
- [ ] No edge-case handling — no salvage parsing, fallbacks, or defensive `try/except`; failures fail loud (`AGENTS.md`)
- [ ] Anonymization / safety-net changes are covered by tests asserting exact output
- [ ] Docs updated if behavior or config changed
