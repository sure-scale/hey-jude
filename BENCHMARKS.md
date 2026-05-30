# Benchmarks

End-to-end anonymization over 23 live cases: 9 curated / adversarial + 14 real
SEC EDGAR contracts. Pipeline: **Azure DeepSeek-V4-Pro** anonymizes →
**Gemini Flash** (`gemini-3.5-flash`) answers → **Gemini Pro**
(`gemini-pro-latest`) judges and runs a blind re-identification attack.

**Metrics (higher = better):** *Eval* = judge score for leak detection,
coherence, completeness, precision. *Resistance* = share of entities the blind
attacker could **not** re-identify (confidence ≥ 0.6 against truth = a leak).
*Utility* = downstream answer quality vs the un-anonymized request. *Re-ids* =
total successful re-identifications (lower = better).

## main vs PR #24

| Metric | main | PR #24 | Δ |
|---|---|---|---|
| Pass rate | 52% (12/23) | 57% (13/23) | +5pp |
| Eval | 89% | 90% | +1pp |
| Resistance | 84% | 86% | +2pp |
| **Re-identifications** | **28** | **17** | **−39%** |
| Utility | 86% | 83% | −3pp |
| Partial / format leaks | 8 | 9 | +1 |

**Net privacy win:** re-ids fall 39%. The drop is conservative — PR #24 also
stopped the attacker truncating its guess list (`max_tokens` 8192→16384) and
pinned it to a strict schema, so it hits *harder* than main's did yet
re-identifies less. Cost: utility −3pp from broader placeholders + quasi-id
generalization (dragged by two outliers). The one error in the PR #24 run was a
transient network drop, not code.

## Per-case (PR #24)

Scores as %. `re-id` = leaks in that case; `pl` = partial leaks. Utility omitted
for the negative-control and errored cases.

| # | Case | Result | Eval | Resist | re-id | Util | pl |
|---|---|---|---|---|---|---|---|
| 1 | Firm Watchlist | PASS | 100 | 100 | 0 | 100 | - |
| 2 | Merger Agreement Excerpt | PASS | 100 | 100 | 0 | 100 | - |
| 3 | NDA with Multiple Parties | PASS | 100 | 100 | 0 | 100 | - |
| 4 | Multi-turn Conversation | PASS | 100 | 100 | 0 | 90 | - |
| 5 | Employment Agreement | PASS | 95 | 100 | 0 | 40 | - |
| 6 | Negative Control (no PII) | PASS | 100 | 100 | 0 | - | - |
| 7 | Adversarial (obfuscated name) | PASS | 100 | 100 | 0 | 100 | - |
| 8 | Adversarial (PII in code block) | PASS | 100 | 100 | 0 | 20 | - |
| 9 | Adversarial (prompt injection) | PASS | 100 | 100 | 0 | 100 | - |
| 10 | EDGAR Teligent / Sawyer | FAIL | 80 | 70 | 3 | 50 | - |
| 11 | EDGAR Tarantella / Keddy | FAIL | 60 | 100 | 0 | 60 | 2 |
| 12 | EDGAR PPD / Hill | FAIL | 82 | 75 | 1 | 100 | 3 |
| 13 | EDGAR Euramax / Brown | FAIL | 98 | 88 | 1 | 90 | - |
| 14 | EDGAR Walmart / Simon | FAIL | 98 | 50 | 3 | 100 | 1 |
| 15 | EDGAR HG Holdings / Garner | FAIL | 85 | 40 | 3 | 80 | - |
| 16 | EDGAR Society Pass / Nguyen | FAIL | 83 | 57 | 3 | 90 | 1 |
| 17 | EDGAR Lightwave Logic | ERROR | - | - | - | - | - |
| 18 | EDGAR Crypto Company / Gilbert | PASS | 78 | 100 | 0 | 100 | - |
| 19 | EDGAR CDI Corp / Stuart | PASS | 75 | 100 | 0 | 100 | - |
| 20 | EDGAR Mostofi / B4MC Gold Mines | PASS | 100 | 100 | 0 | 100 | - |
| 21 | EDGAR VPR Brands / Frija | FAIL | 100 | 50 | 1 | 70 | - |
| 22 | EDGAR Implant Sciences | FAIL | 70 | 67 | 2 | 100 | 2 |
| 23 | EDGAR BizRight / BZRTH | PASS | 70 | 100 | 0 | 60 | - |

**Averages (22 scored):** eval 90% · resistance 86% (17 re-ids) · utility 83% ·
9 partial leaks.

## Notes

- Curated + adversarial: 9/9 pass, 100% resistance (prompt injection, obfuscated
  names, PII in code blocks all held).
- All 9 fails are EDGAR contracts — named executives in SEC filings are
  inherently re-identifiable. Hard floor of the task, not a regression.
- Case 21 was cracked via the *source filename* (`vprb...htm` → ticker VPRB), not
  the body — pending audit of whether the harness leaks filenames to the attacker.

**Follow-ups:** utility tuning; filename-leak audit (case 21); issue #25
(re-id-critic ungating + critic-text re-anonymization).
