<p align="left">
  <img src="docs/assets/jude-logo.png" alt="Hey Jude logo" width="120">
</p>

# Hey Jude

**Privacy gateway for legal LLM workflows.**

<p align="left">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="Docker Compose" src="https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white">
  <img alt="OpenAI-compatible" src="https://img.shields.io/badge/OpenAI--compatible-API-111827?style=flat-square">
  <img alt="License AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-111827?style=flat-square">
</p>

Hey Jude is a drop-in proxy that sits between your app and your LLM provider. It strips PII out of prompts before they leave your environment and restores it in the response — your users see real names, the cloud LLM never does. Point any OpenAI, Anthropic, or Gemini SDK at it and keep your existing code.

```
You send:   "Draft a demand letter for John Doe v. Goldman Sachs."
LLM sees:   "Draft a demand letter for PERSON_01 v. COMPANY_01."
You get back: a letter naming John Doe and Goldman Sachs.
```

A local LLM does the swap, so it understands context: legal defined terms like "the Purchaser" stay intact while real names, emails, and addresses become semantic placeholders that keep enough meaning for the model to reason well. A Presidio safety net catches anything it misses.

This is a helper layer for data minimization, not a guarantee. Use it as part of a broader confidentiality strategy.

---

## How It Works

1. Your app sends a chat completion request to Hey Jude (OpenAI-compatible API).
2. A local LLM analyzes the text and identifies real PII vs. legal/structural terms.
3. PII gets replaced with semantic placeholders. Legal defined terms are kept.
4. A Presidio safety net scans the result for anything the LLM missed.
5. On high-sensitivity requests, a re-identification critic checks the sanitized text and broadens any placeholder a blind attacker could still pin down.
6. The request is routed by residual risk: pseudonymized prompts go to your chosen provider; requests whose PII cannot be reduced below the re-identification threshold are handled per your `IRREDUCIBLE_POLICY` (block, ask, route to a sovereign in-jurisdiction model, or allow).
7. The response comes back, placeholders are swapped for originals, and your app gets a normal-looking reply.

---

## Why It Exists

*   **Context-aware anonymization:** A local LLM understands that "Goldman Sachs" is PII but "the Purchaser" is a legal term — something regex and NER can't do reliably.
*   **Semantic placeholders:** typed tokens like `PERSON_01` and `COMPANY_01`, not opaque `ENTITY_01`. The downstream LLM keeps the structural role it needs to reason, while the category stays broad enough not to re-identify (see Inference Resistance).
*   **Safety net:** Presidio runs after the LLM as a second pass. Configurable as `warn` (auto-fix), `strict` (reject), or `off`.
*   **Inference resistance:** Stripping the literal name is not enough when context still re-identifies it. A re-identification critic runs a blind attack on the sanitized output and broadens any placeholder that gives the entity away.
*   **Jurisdiction-aware routing:** Some requests carry PII that cannot be masked below the re-identification threshold. Hey Jude classifies that residue and routes it by policy — including to a sovereign in-jurisdiction model — instead of silently sending it to a US frontier provider.
*   **Drop-in API:** Exposes OpenAI, Anthropic, and Gemini-compatible endpoints. Point existing SDKs at the gateway.
*   **Fully local by default:** Runs without cloud keys using Ollama for both anonymization and demo responses.
*   **Cloud routing when ready:** Route anonymized prompts to OpenAI, Anthropic, Gemini, Azure, or any LiteLLM-compatible provider.

---

## Quick Start

### 1. Install Ollama and Pull the Default Model

```bash
ollama pull qwen3.5:4b
```

### 2. Run the Gateway

```bash
git clone https://github.com/nickwatson/hey-jude.git
cd hey-jude
cp .env.example .env
docker compose up --build
```

The gateway will run at `http://localhost:4005`.

### 3. Test It

In another terminal:

```bash
python3 tests/e2e/test_gateway.py
```

The default setup is fully local: Redis runs in Docker, and Ollama runs on your host machine.

---

## Default Configuration

The defaults work out of the box for most users.

| Variable | Default | Purpose |
|----------|---------|---------|
| `REDIS_URL` | `redis://localhost:6379/0` | Temporary mapping storage |
| `API_KEY` | `sk-heyjude-dev` | Gateway authentication |
| `LOCAL_LLM_URL` | `http://localhost:11434/v1` | Local anonymization endpoint |
| `LOCAL_LLM_MODEL` | `qwen3.5:4b` | Local anonymization model |
| `LOCAL_LLM_API_KEY` | *(empty)* | API key for cloud-hosted anonymization models |
| `EXTERNAL_LLM_MODEL` | `ollama_chat/qwen3.5:4b` | Destination model via LiteLLM |
| `EXTERNAL_LLM_API_BASE` | `http://localhost:11434` | LiteLLM API base for Ollama |
| `EXTERNAL_LLM_MODEL_SENSITIVE` | `ollama_chat/qwen3.5:4b` | In-jurisdiction sovereign destination for irreducible-PII requests (self-hosted OSS, not subject to US compulsion) |
| `EXTERNAL_LLM_MODEL_SENSITIVE_API_BASE` | `http://localhost:11434` | LiteLLM API base for the sovereign destination |
| `IRREDUCIBLE_POLICY` | `WARN` | Policy for irreducible PII: `BLOCK` refuses, `ASK` returns a confirmation round-trip, `WARN` routes to the sovereign destination and flags it, `ALLOW` sends to the US frontier model anyway. Per-request override via the `X-Heyjude-Policy` header |
| `ANONYMIZATION_MODE` | `llm` | `llm` (context-aware) or `mechanical` (NER-only) |
| `SAFETY_NET_STRICTNESS` | `warn` | `warn` (auto-fix), `strict` (reject), or `off` |
| `REID_CRITIC_ENABLED` | `true` | Run a blind re-identification pass on high-sensitivity output and broaden placeholders the critic cracks |
| `DOCUMENT_UNREADABLE_ACTION` | `reject` | What to do when an uploaded file has no readable text layer: `reject`, `warn`, or `skip` |
| `CUSTOM_RECOGNIZERS_PATH` | *(unset)* | Path to a YAML/JSON file of custom Presidio regex recognizers |
| `KNOWN_ENTITIES_PATH` | *(unset)* | Path to a YAML/JSON known-entity dictionary |
| `AUDIT_ENABLED` | `false` | Enable request-level audit logging |
| `AUDIT_DESTINATION` | `stdout` | `stdout` or a file path |
| `AUDIT_CONTENT_LEVEL` | `metadata` | `metadata` (digests only), `anonymized` (PII-free payload), `full` (raw content) |
| `AUDIT_ROTATION` | `monthly` | Segment files by period: `none`, `daily`, `monthly` |
| `AUDIT_FAILURE_MODE` | `ignore` | `ignore` (logging never blocks a request) or `fail` (fail-closed) |

When running through Docker Compose, the service automatically uses `host.docker.internal` so the container can reach Ollama on your Mac.

### Choosing Models

Hey Jude uses **two separate models** for two different jobs, and you size them independently:

| Role | Setting | Job | Pick for |
|------|---------|-----|----------|
| Anonymizer | `LOCAL_LLM_MODEL` (+ `LOCAL_LLM_URL`) | Classify entities and emit placeholder JSON | Speed and cost — a small, fast model is enough |
| Destination | `EXTERNAL_LLM_MODEL` | Do the actual legal work on the already-anonymized prompt | Capability — your strongest model |

The anonymizer's task is narrow and structured (find PII, output a fixed JSON schema), so it does **not** need a frontier model. Putting the cheapest model that holds classification quality here cuts cost and latency on every request, because the anonymizer runs once per message before anything reaches the destination. Reserve the expensive, capable model for `EXTERNAL_LLM_MODEL`, which never sees raw PII anyway.

The two run on independent endpoints and providers — e.g. a small Azure or Ollama model for anonymization and Gemini, Anthropic, or OpenAI for the destination — so you tune the cost/quality trade-off on each without touching the other.

### Domain-Specific Detection

Default NER misses the abbreviated, inconsistent names common in legal text ("Call w/ J. Smith re: Acme merger"). Two opt-in mechanisms close the gap. Templates live in [`examples/`](examples/).

**Custom recognizers** (`CUSTOM_RECOGNIZERS_PATH`) add regex-based entity types — matter numbers, client codes, opposing-counsel formats. They run in the Presidio safety net and as a mechanical-mode detection strategy.

**Known-entity dictionary** (`KNOWN_ENTITIES_PATH`) is a firm-maintained list of the names you must never leak — clients, personnel, matter names. Listed entities are matched case-insensitively and **guaranteed replaced before the prompt reaches the LLM**, so a critical name never depends on the model noticing it. All spelling variants (`term` + `aliases`) collapse to one placeholder.

By default an auto-numbered placeholder (e.g. `CLIENT_NAME_01`) is assigned per request. Set `replace_with` on an entry to fix its placeholder so it stays identical across every request.

Hey Jude extracts text from common legal document formats before anonymization, including text PDFs, DOCX, HTML, EML, TXT, Markdown, and RTF. Scanned PDFs, flattened PDFs, and images are not OCRed yet; by default they are rejected so unreadable content is not forwarded without anonymization.

### Inference Resistance

Removing the literal name does not always anonymize. A placeholder like `CLOUD_COMPUTING_COMPANY_01` next to a location, a job title, and a few figures can name the entity as plainly as the original string. Hey Jude defends against this on two levels:

*   **Broad placeholders.** The anonymizer is instructed to pick the broadest category that still supports the task, and to broaden further until a reader cannot name the entity.
*   **Quasi-identifier generalization.** On high-sensitivity requests, precise non-PII figures that re-identify in combination — an exact revenue, deal value, filing date, headcount, or market share — are generalized to a qualitative band (`$450,000` → "a high-six-figure sum", `March 15, 2024` → "early 2024") that preserves the analytic role without the identifying digits.
*   **Minimal descriptors.** The context handed to the downstream model is kept to the minimum needed to stay readable ("a company", "a senior executive") and never a distinguishing detail.
*   **Re-identification critic.** On high-sensitivity requests (`REID_CRITIC_ENABLED`, on by default), one extra local-LLM pass runs a blind re-identification attack over the sanitized output — it sees only the placeholders and descriptors, never the original or the mapping — and broadens the descriptor of any entity it pins down with confidence at or above `REID_CRITIC_THRESHOLD` (default `0.6`). It is bounded to a single call and best-effort: it never fails the request.

The end-to-end benchmark scores this dimension explicitly with a blind attacker; see [BENCHMARKS.md](BENCHMARKS.md).

### Jurisdiction-Aware Routing

Some requests carry PII that cannot be pseudonymized below a meaningful re-identification threshold — quasi-identifiers re-identify, and "singling out" survives pseudonymization. Sending that residue to a US frontier provider is the wrong default under a CLOUD Act / FISA 702 threat model (lawful compulsion of a US provider, not a breach).

The anonymizer classifies whether a request is **irreducible** independently of whether masking succeeded. `IRREDUCIBLE_POLICY` then decides what happens:

| Policy | Behavior |
|--------|----------|
| `BLOCK` | Refuse the request (403) before any egress |
| `ASK` | Return a confirmation round-trip before routing |
| `WARN` (default) | Route to the sovereign in-jurisdiction destination (`EXTERNAL_LLM_MODEL_SENSITIVE`) and flag it |
| `ALLOW` | Send to the US frontier model anyway |

Reducible requests route exactly as before — sensitivity alone never diverts. Only an irreducible classification triggers the sovereign path. Override per request with the `X-Heyjude-Policy` header (invalid values return 400). The chosen `route_tier` and the `irreducible` flag are recorded in the audit log, so any silent-US-egress is auditable after the fact. Note that Azure OpenAI and AWS Bedrock are **not** sovereign tiers — the CLOUD Act reaches the US parent regardless of the residency region.

### Audit Logging

Set `AUDIT_ENABLED=true` to record one envelope per request: timestamps, latency, which external model it was routed to, entity count, sensitivity, safety-net result, the per-entity anonymization decisions, and SHA-256 digests of the input and the anonymized output. This is the artifact that proves anonymization happened and that only PII-free content left the network.

**Per-entity decisions.** In LLM mode each record carries a `decisions` list — what the anonymizer found and what it did to it (`action` is `replace`, `keep`, or `generalize`) with the reason. At the default `metadata` level this stores `entity_type`, `action`, and `reason` only, never the raw entity text; `full` additionally records the original text and its replacement. This is the per-matter "what did we send, what did we withhold, why" trail for discovery and malpractice defense.

**Tamper-evident.** The log is hash-chained JSONL: each record carries the hash of the previous one, so editing or deleting any historical record breaks the chain from that point on — detectable even by someone with write access. Verify a segment at any time:

```bash
hey-jude audit verify audit/audit-2026-05.jsonl
```

Set `AUDIT_HMAC_KEY` to bind the chain to a secret so an attacker who cannot read the key cannot recompute valid hashes. Walk a segment for conflict checks, client audits, or discovery production:

```bash
hey-jude audit query audit/audit-2026-05.jsonl --matter M-123456 --since 2026-05-01
```

**Content level.** The default `metadata` stores **no raw client PII** — only digests — so the audit trail itself does not become a confidential-data store. Choose `anonymized` to retain the PII-free payload (useful as a malpractice-defensible record of what the AI was actually asked and answered, without storing client identities). `full` additionally persists the raw pre-anonymization content and is a deliberate PII honeypot; it logs a startup warning and should be reserved for environments where that risk is understood. Tag requests with the `X-Heyjude-Matter-Id` header so records are queryable by matter; enable `AUDIT_ACTOR_HEADER` only if your firm policy permits attorney attribution.

**Immutability vs. retention.** A hash chain makes history immutable, but legal duties (matter-close destruction, data-subject erasure, retention schedules) require eventual deletion. Hey Jude resolves this with period segments (`AUDIT_ROTATION`): each month/day is an independent chain in its own file, so an expired segment can be destroyed wholesale without invalidating the active chain. **Suspend rotation and deletion while a matter is under legal hold.** For cryptographic-grade WORM, point `AUDIT_DESTINATION` at a write-once volume (`chattr +a` on Linux) or ship sealed segments to object storage with an immutability lock (e.g. S3 Object Lock). Keep the log on encrypted disk; Hey Jude does not encrypt records itself.

---

## Native Python

If you prefer not to use Docker, run Redis yourself and start the app directly:

```bash
pip install -e ".[dev]"
python3 -m spacy download en_core_web_lg
uvicorn hey_jude.main:app --host 0.0.0.0 --port 4005
```

---

## Advanced Models

The default `qwen3.5:4b` target is chosen for low-friction local setup. If you want a different local model, pull it with Ollama and update `LOCAL_LLM_MODEL` plus `EXTERNAL_LLM_MODEL`.

| Use case | Model |
|----------|-------|
| Fastest tiny local test | `qwen3.5:0.8b` |
| Default local setup | `qwen3.5:4b` |
| Stronger local setup | `qwen3.5:9b` |
| High-end local setup | `qwen3.6:35b-a3b` |

Example:

```bash
ollama pull qwen3.5:9b
LOCAL_LLM_MODEL=qwen3.5:9b
EXTERNAL_LLM_MODEL=ollama_chat/qwen3.5:9b
```

To use Apple MLX instead of Ollama, serve an OpenAI-compatible endpoint and point `LOCAL_LLM_URL` or `EXTERNAL_LLM_API_BASE` at that server.

To route the final prompt to a cloud provider, set `EXTERNAL_LLM_MODEL` to any LiteLLM model identifier and provide that provider's API key, for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`.

### Azure AI as Anonymization Backend

Instead of running Ollama locally, you can use an Azure AI-hosted model for the anonymization layer. This is useful on machines where local inference is slow (e.g., laptops without dedicated GPU).

Azure AI Foundry exposes OpenAI-compatible endpoints for models deployed from the model catalog. Set these in your `.env`:

```bash
LOCAL_LLM_URL=https://<your-resource>.openai.azure.com/openai/v1
LOCAL_LLM_MODEL=DeepSeek-V4-Pro
LOCAL_LLM_API_KEY=<your-azure-api-key>
```

Available models (same endpoint, swap `LOCAL_LLM_MODEL`):

| Model | `LOCAL_LLM_MODEL` value | Notes |
|-------|------------------------|-------|
| DeepSeek V4 Pro | `DeepSeek-V4-Pro` | Recommended — fast, strong at structured JSON output |
| Kimi K2.6 | `Kimi-K2.6` | Reasoning model, needs high `max_tokens` (uses thinking tokens) |

Any model deployed to your Azure AI project that serves an OpenAI-compatible chat completions endpoint will work. The gateway sends requests to `{LOCAL_LLM_URL}/chat/completions` with both `api-key` and `Authorization: Bearer` headers.

**Prompt caching.** The anonymization prompt (`prompts/anonymize.txt`) keeps its large static block — task, classification instructions, and output schema — first, and the per-request variables (the existing placeholder mapping and the message text) last. That fixed prefix is identical on every request, so providers with automatic prompt caching (Azure OpenAI, Anthropic, Gemini) reuse it instead of re-billing the instructions each call, cutting input-token cost and latency on the hot path. If you edit the template, keep the variables at the end or the cached prefix is lost.

### E2E Testing

The end-to-end test uses three models:

| Role | Model | Purpose |
|------|-------|---------|
| Anonymizer | Configured via `LOCAL_LLM_*` | PII detection and replacement |
| Destination | Gemini Flash | Receives anonymized prompts |
| Evaluator | Gemini Pro | Judges anonymization quality (PII leaks, coherence, completeness) |

```bash
GEMINI_API_KEY=your-key python3 tests/e2e/test_gemini_anonymization.py
```

The test auto-downloads public-domain legal documents from SEC EDGAR on first run (NDAs, employment agreements, settlement agreements, etc.) and uses them alongside inline test cases. Downloaded documents are cached locally and gitignored.

Beyond literal-leak checks, the harness runs a blind re-identification attacker, a utility-preservation judge, and a two-track score that separates reducible from irreducible requests. Headline results and methodology are in [BENCHMARKS.md](BENCHMARKS.md).

---

## SDK Integration

Since Hey Jude behaves like standard LLM endpoints, existing clients can point at the gateway.

### Mike OSS

Mike can route OpenAI, Claude, and Gemini calls through Hey Jude. After starting this gateway, set in Mike's `backend/.env`:

```bash
HEY_JUDE_ENABLED=true
HEY_JUDE_BASE_URL=http://localhost:4005
HEY_JUDE_API_KEY=sk-heyjude-dev
```

### Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4005/v1",
    api_key="sk-heyjude-dev",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "I am John Doe and I work at Google."}
    ],
)
print(response.choices[0].message.content)

response = client.responses.create(
    model="gpt-4o",
    input="I am John Doe and I work at Google.",
)
print(response.output_text)
```

### Python Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:4005",
    api_key="sk-heyjude-dev",
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "I am John Doe and I work at Google."}
    ],
)
print(response.content[0].text)
```

### Node.js OpenAI SDK

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:4005/v1',
  apiKey: 'sk-heyjude-dev',
});

const response = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'I am John Doe and I work at Google.' }],
});
console.log(response.choices[0].message.content);

const responsesResponse = await client.responses.create({
  model: 'gpt-4o',
  input: 'I am John Doe and I work at Google.',
});
console.log(responsesResponse.output_text);
```

### Node.js Anthropic SDK

```javascript
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic({
  baseURL: 'http://localhost:4005',
  apiKey: 'sk-heyjude-dev',
});

const response = await client.messages.create({
  model: 'claude-3-5-sonnet-20241022',
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'I am John Doe and I work at Google.' }],
});
console.log(response.content[0].text);
```

---

## Contributing & Security

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the no-edge-case / fail-loud law, and PR conventions. Use synthetic data only in issues, tests, and PRs — never real PII, secrets, or keys.

Report suspected vulnerabilities or PII leaks privately via a GitHub security advisory, not a public issue. See [SECURITY.md](SECURITY.md).

---

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
