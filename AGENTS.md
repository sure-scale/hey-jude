# AGENTS.md

<law severity="absolute" violations="zero-tolerance">
NO edge-case handling in prompts or tests, ever.

No salvage parsing, no retry loops, no fallback branches, no defensive
try/except, no "tolerate malformed output", no lenient coercion, no graceful
degradation — not in any prompt, and not in any test.

Prompts state the contract once, plainly. Tests assert exact behavior and
nothing else. When input is malformed or a call fails, FAIL LOUD. A failure is
a signal to fix the root cause in production code, never to paper over in the
prompt or the test.

This rule has no exceptions. "Just this once", "only transient", "it's
harmless", and "the model sometimes does X" are all rejected.
</law>
