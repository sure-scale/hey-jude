# Benchmarks

End-to-end anonymization over 34 live cases: 9 curated / adversarial + a real
SEC EDGAR corpus run on **two tracks** (a generic-review prompt and an
identity-essential diligence prompt per contract). Pipeline: **Azure
DeepSeek-V4-Pro** anonymizes → **Gemini Flash** (`gemini-3.5-flash`) answers →
**Gemini Pro** (`gemini-pro-latest`) judges and runs a blind re-identification
attack.

**Metrics (higher = better):** *Eval* = judge score for leak detection,
coherence, completeness, precision. *Resistance* = share of entities the blind
attacker could **not** re-identify (confidence ≥ 0.6 against truth = a leak).
*Utility* = downstream answer quality vs the un-anonymized request. *Re-ids* =
total successful re-identifications (lower = better).

## Two-track scoring

A single pass-rate is **mathematically capped** while the corpus contains cases
whose `entity × sensitive-attribute` link is essential to the answer — they
cannot be made non-reconstructable (quasi-identifiers re-identify; "singling
out" survives pseudonymization). Scoring every case on "not-reconstructable"
reads that theoretical floor as a regression. Each case carries an
**independently-authored** `expected_class`:

- **Reducible** — masking alone prevents identification. Scored as before (any
  literal / partial / inferred leak → fail).
- **Irreducible** — re-id is the *expected floor*, not a failure. Success = the
  pipeline **recognized** irreducibility and **routed it to a sovereign tier**.
  The worst case is the **silent mishandle**: an irreducible request treated as
  reducible (no flag → would egress to a US tier).

**Reducibility is a property of the request, not the document.** The same EDGAR
contract is *reducible* under "review this agreement" (masking satisfies the
ask) and *irreducible* under "background diligence on <the named executive>"
(`NAMED_ENTITY_ESSENTIAL` — the question demands the real person). Each contract
with a real named subject therefore runs on both tracks. This is a claim about
the *nature* of each prompt, never a relabelling of "it failed".

## Run of record

Headline numbers from the latest live run (34 cases):

| Track | Result |
|---|---|
| **Irreducible** | **11/11 (100%)** — detection 100%, **silent-mishandle 0**, silent-US-egress 0/11 |
| Reducible | 16/22 (73%) |
| Over-flag (reducible flagged irreducible) | 0/22 (0%) |
| **Overall** | **27/34 (79%)** — 6 failed, 1 errored |

| Aggregate | Score |
|---|---|
| Eval | 89% |
| Resistance | 93% (15 re-ids across 33 cases) |
| Utility | 51% |
| Partial / format leaks | 19 |

The irreducible track is the issue-#27 deliverable, and it is clean: every
identity-essential request was recognized and diverted to the sovereign tier,
**zero silent mishandles, zero silent US egress**. The reducible track's misses
and the one error are honest residue (below), not regressions — they are kept as
real numbers per the no-papering-over rule.

### Nondeterminism

The anonymizer model is non-deterministic on borderline inputs. Across runs the
irreducibility **detection rate sits at 91–100%** and the **silent-mishandle
count can be nonzero** (a borderline diligence prompt occasionally classified
reducible). The run of record shows detection 100% / silent-mishandle 0; a prior
run showed 91% / 1. Treat the headline as a band, not a fixed point — the
benchmark is not re-rolled for a favorable draw.

## Jurisdiction tiers

The threat is **lawful compulsion of a US provider** (CLOUD Act / FISA 702), not
a breach. Route destinations are tiered by *actual jurisdiction*, not by
"managed = safe":

| Tier | Destination | Sovereign? |
|---|---|---|
| 1 | Public US API (frontier) | No — US jurisdiction |
| 2 | US hyperscaler (Azure OpenAI / AWS Bedrock) | **No** — CLOUD Act reaches the US parent regardless of data-residency region |
| 3 | EU-sovereign managed | Partial |
| 4 | Self-hosted OSS in-jurisdiction | **Yes** (gold standard) |

The pipeline ships **two** tiers today: `us-ok-pseudonymized` (Tier 1, pseudonyms
only) and `in-jurisdiction-sensitive` (Tier 4, sovereign). Treating Azure/Bedrock
as sovereign is a false-safety failure — they are **not** in `SOVEREIGN_TIERS`.
The pseudonym↔real mapping stays in the firm-deployed Redis, so a production
order against the US provider yields only pseudonyms.

## Per-case

Scores as %. `re-id` = leaks in that case; `pl` = partial leaks; Track R =
reducible, I = irreducible. Utility omitted for the negative-control case; the
irreducible track's utility is **expected** to collapse (the sovereign-routed
answer deliberately withholds the real-person diligence the prompt demanded).

| # | Track | Case | Result | Eval | Resist | re-id | Util | pl |
|---|---|---|---|---|---|---|---|---|
| 1 | R | Firm Watchlist | PASS | 100 | 100 | 0 | 100 | - |
| 2 | R | Merger Agreement Excerpt | PASS | 98 | 100 | 0 | 90 | - |
| 3 | R | NDA with Multiple Parties | PASS | 100 | 100 | 0 | 100 | - |
| 4 | R | Multi-turn Conversation | PASS | 98 | 100 | 0 | 70 | - |
| 5 | R | Employment Agreement | PASS | 95 | 100 | 0 | 40 | - |
| 6 | R | Negative Control (no PII) | PASS | 100 | 100 | 0 | - | - |
| 7 | R | Adversarial (obfuscated name) | PASS | 100 | 100 | 0 | 100 | - |
| 8 | R | Adversarial (PII in code block) | PASS | 100 | 100 | 0 | 10 | - |
| 9 | R | Adversarial (prompt injection) | PASS | 100 | 100 | 0 | 100 | - |
| 10 | R | EDGAR Teligent / Sawyer | PASS | 85 | 100 | 0 | 80 | - |
| 11 | I | EDGAR·ID Sawyer diligence | PASS | 73 | 60 | 4 | 0 | 1 |
| 12 | R | EDGAR Tarantella / Keddy | FAIL | - | 100 | 0 | 30 | 2 |
| 13 | I | EDGAR·ID Keddy diligence | PASS | - | 100 | 0 | 0 | 2 |
| 14 | R | EDGAR PPD / Hill | FAIL | 75 | 60 | 2 | 100 | 3 |
| 15 | I | EDGAR·ID Hill diligence | PASS | 93 | 33 | 4 | 0 | 2 |
| 16 | R | EDGAR Euramax / Brown | PASS | 100 | 100 | 0 | 100 | - |
| 17 | I | EDGAR·ID Brown diligence | PASS | 75 | 100 | 0 | 0 | 1 |
| 18 | R | EDGAR Walmart / Simon | FAIL | 98 | 100 | 0 | 100 | 1 |
| 19 | I | EDGAR·ID Simon diligence | PASS | 90 | 50 | 3 | 0 | 1 |
| 20 | R | EDGAR HG Holdings / Garner | FAIL | 75 | 90 | 1 | 40 | 1 |
| 21 | I | EDGAR·ID Garner diligence | PASS | 95 | 100 | 0 | 0 | - |
| 22 | R | EDGAR Society Pass / Nguyen | FAIL | 88 | 100 | 0 | 80 | 1 |
| 23 | I | EDGAR·ID Nguyen diligence | PASS | 85 | 100 | 0 | 0 | 1 |
| 24 | R | EDGAR Lightwave Logic (template) | ERROR | - | - | - | - | - |
| 25 | R | EDGAR Crypto Company / Gilbert | PASS | 65 | 100 | 0 | 100 | - |
| 26 | I | EDGAR·ID Gilbert diligence | PASS | 93 | 100 | 0 | 0 | - |
| 27 | R | EDGAR CDI Corp / Stuart | PASS | 68 | 100 | 0 | 90 | - |
| 28 | I | EDGAR·ID Stuart diligence | PASS | - | 100 | 0 | 0 | - |
| 29 | R | EDGAR Mostofi / B4MC Gold Mines | PASS | 88 | 100 | 0 | 50 | - |
| 30 | R | EDGAR VPR Brands / Frija | PASS | 98 | 100 | 0 | 100 | - |
| 31 | I | EDGAR·ID Frija diligence | PASS | 88 | 100 | 0 | 0 | - |
| 32 | R | EDGAR Implant Sciences | FAIL | 73 | 83 | 1 | 100 | 2 |
| 33 | R | EDGAR BizRight / BZRTH | PASS | 75 | 100 | 0 | 40 | - |
| 34 | I | EDGAR·ID Huang diligence | PASS | 85 | 100 | 0 | 0 | 1 |

## Notes

- Curated + adversarial: 9/9 pass, 100% resistance (prompt injection, obfuscated
  names, PII in code blocks all held).
- **The one error is the validator failing loud, not a leak.** Lightwave Logic is
  a blank fill-in-the-blank NDA template; the model emitted a self-referential
  mapping (`the Company` → `the Company`) and the leak-check refused it
  (`ValueError`). Erroring instead of emitting an identity-replacement is the
  desired safety behavior on a degenerate input.
- The reducible-track fails are genuine local-model **recall limits**, kept as
  real numbers: multi-token surface fragments (Tarantella "Santa Clara County
  Superior Court"; Implant address + "US patent 6,183,409"), contextual
  inference (PPD CRO + 2011; Walmart "associate"), and descriptor
  over-disclosure (BizRight "a large online retailer" → Amazon). These reflect
  the anonymizer's recall, not an issue-#27 regression.
- The irreducible track's utility is **0% by design** — the request needs the
  real named person, the sovereign route withholds it. That collapse is the
  honest cost the two-tier policy is built to surface, not a defect.

**Follow-ups:** reducible-track recall tuning (multi-token surface forms,
contextual inference); issue #25 (re-id-critic ungating + critic-text
re-anonymization).
