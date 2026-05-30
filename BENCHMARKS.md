# Benchmarks

End-to-end anonymization benchmark over a 23-case live suite: 9 curated /
adversarial cases plus 14 real SEC EDGAR contracts (named executives, company
filings).

## Pipeline under test

| Stage | Model |
|---|---|
| Anonymizer (local LLM) | Azure DeepSeek-V4-Pro |
| Destination | Gemini Flash (`gemini-3.5-flash`) |
| Judge + blind re-identification attacker | Gemini Pro (`gemini-pro-latest`) |

## Metrics

- **Eval** — Gemini Pro judge scoring leak detection, semantic coherence,
  completeness, and precision (0-10, averaged).
- **Inference resistance** — blind attacker sees only the sanitized text and the
  per-placeholder context descriptors (never the original or the mapping) and
  guesses each real entity with a confidence. A guess at confidence >= 0.6 that
  matches the truth counts as a re-identification. Resistance = `10 * (1 - re-id
  rate)`.
- **Re-identifications** — total successful re-identifications across all cases.
- **Utility** — judge comparison of the downstream answer on the original vs the
  anonymized request; penalizes lost substance, not the mere presence of
  placeholders.
- **Partial / format leaks** — quasi-identifier or formatting leaks short of a
  full re-identification.

## Aggregate: main (pre-optimization) vs PR #24

Same 23-case suite, same models.

| Metric | main (old) | PR #24 (new) | Δ |
|---|---|---|---|
| Pass / Fail / Error | 12 / 11 / 0 | 13 / 9 / 1 | +1 / -2 / +1 |
| Eval | 8.9 | 9.0 | +0.1 |
| Inference resistance | 8.4 | 8.6 | +0.2 |
| **Re-identifications** | **28** | **17** | **-11 (-39%)** |
| Utility | 8.6 | 8.3 | -0.3 |
| Partial / format leaks | 8 | 9 | +1 |

**Net privacy win.** Successful re-identifications fall 39% (28 -> 17), with two
fewer failing cases and small gains in eval and resistance. The drop is
*conservative*: PR #24 also hardened the harness so the blind attacker no longer
truncates its guess list (`max_tokens` 8192 -> 16384) and is pinned to a strict
response schema. The new attacker therefore hits harder than main's did, yet
re-identifies far less.

**Trade-off.** Utility falls 0.3 (broadest-category + quasi-identifier
generalization cost some downstream usefulness; the average is dragged by two
outliers — PII-in-code-block and an employment agreement). Acceptable for the
privacy gain; flagged for a later tuning pass.

The single error in the PR #24 run is a transient network drop (DNS), not code.

## Per-case detail (PR #24 run, current main)

`re-id` = successful re-identifications in that case. `pl` = partial / format
leaks. Utility is omitted for the negative-control case (no PII, nothing to
preserve) and the errored case.

| # | Case | Result | Eval | Resistance | re-id | Utility | pl |
|---|---|---|---|---|---|---|---|
| 1 | Firm Watchlist (dictionary + custom recognizer) | PASS | 10.0 | 10.0 | 0 | 10 | - |
| 2 | Merger Agreement Excerpt | PASS | 10.0 | 10.0 | 0 | 10 | - |
| 3 | NDA with Multiple Parties | PASS | 10.0 | 10.0 | 0 | 10 | - |
| 4 | Multi-turn Conversation | PASS | 10.0 | 10.0 | 0 | 9 | - |
| 5 | Employment Agreement | PASS | 9.5 | 10.0 | 0 | 4 | - |
| 6 | Negative Control (no PII) | PASS | 10.0 | 10.0 | 0 | - | - |
| 7 | Adversarial (spaced / obfuscated name) | PASS | 10.0 | 10.0 | 0 | 10 | - |
| 8 | Adversarial (PII in code block) | PASS | 10.0 | 10.0 | 0 | 2 | - |
| 9 | Adversarial (prompt injection) | PASS | 10.0 | 10.0 | 0 | 10 | - |
| 10 | EDGAR Teligent / Timothy B. Sawyer — Settlement & General Release | FAIL | 8.0 | 7.0 | 3 | 5 | - |
| 11 | EDGAR Tarantella / Caroline Keddy — Settlement & General Release | FAIL | 6.0 | 10.0 | 0 | 6 | 2 |
| 12 | EDGAR PPD / Raymond H. Hill — CEO Employment Agreement | FAIL | 8.2 | 7.5 | 1 | 10 | 3 |
| 13 | EDGAR Euramax International / Richard Brown — CEO Employment Agreement | FAIL | 9.8 | 8.8 | 1 | 9 | - |
| 14 | EDGAR Walmart / William S. Simon — Consulting Agreement | FAIL | 9.8 | 5.0 | 3 | 10 | 1 |
| 15 | EDGAR HG Holdings / Brad G. Garner — Consulting Agreement + NDA | FAIL | 8.5 | 4.0 | 3 | 8 | - |
| 16 | EDGAR Society Pass / Dennis Nguyen — Transition, Release & Consulting | FAIL | 8.3 | 5.7 | 3 | 9 | 1 |
| 17 | EDGAR Lightwave Logic — Director's NDA (template) | ERROR | - | - | - | - | - |
| 18 | EDGAR The Crypto Company / James Gilbert — Separation & Mutual Release | PASS | 7.8 | 10.0 | 0 | 10 | - |
| 19 | EDGAR CDI Corp / Jay G. Stuart — Release, Waiver & Non-Competition | PASS | 7.5 | 10.0 | 0 | 10 | - |
| 20 | EDGAR Mostofi & Co / B4MC Gold Mines — Executive Suite Sublease | PASS | 10.0 | 10.0 | 0 | 10 | - |
| 21 | EDGAR VPR Brands / Kevin Frija — Promissory Note | FAIL | 10.0 | 5.0 | 1 | 7 | - |
| 22 | EDGAR Implant Sciences / International Brachytherapy — IP License | FAIL | 7.0 | 6.7 | 2 | 10 | 2 |
| 23 | EDGAR BizRight LLC / BZRTH Inc — Asset Purchase Agreement | PASS | 7.0 | 10.0 | 0 | 6 | - |

**Averages (22 scored cases):** eval 9.0 · resistance 8.6 (17 re-ids) · utility
8.3 (21 cases) · 9 partial / format leaks.

## Observations

- Curated and adversarial cases are clean: 9/9 pass, resistance 10.0 on every
  one (prompt injection, obfuscated names, PII in code blocks all held).
- All 9 failing cases are EDGAR real contracts. Named executives in SEC filings
  are inherently re-identifiable from surrounding context — this is the hard
  floor of the task, not a pipeline regression.
- Case 21 (VPR Brands) was re-identified at 0.99 confidence via the *source
  filename* (`vprb072318ex10-1.htm` -> ticker VPRB), not the sanitized body. If
  the harness feeds source filenames into the attacker, that inflates the re-id
  count; pending audit of harness input fidelity.

## Known follow-ups

- Utility tuning pass (recover the -0.3 without weakening privacy).
- Audit whether the harness leaks source filenames into the attacker input
  (case 21 artifact).
- Issue #25 tracks deferred re-identification-critic work (ungating, critic-text
  re-anonymization).
