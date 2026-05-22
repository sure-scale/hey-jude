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

Hey Jude sits between your app and your LLM provider. It strips PII from prompts before they leave your environment, then restores the original details in the response. Your users see real names; the cloud LLM never does.

It uses a local LLM to understand context — so legal defined terms like "the Purchaser" stay intact while real names, emails, and addresses get replaced with semantic placeholders like `INVESTMENT_BANK_01` or `PERSON_02`. A Presidio-based safety net catches anything the LLM misses.

This is a helper layer for data minimization, not a guarantee. Use it as part of a broader confidentiality strategy.

---

## How It Works

1. Your app sends a chat completion request to Hey Jude (OpenAI-compatible API).
2. A local LLM analyzes the text and identifies real PII vs. legal/structural terms.
3. PII gets replaced with semantic placeholders. Legal defined terms are kept.
4. A Presidio safety net scans the result for anything the LLM missed.
5. The sanitized prompt is forwarded to your chosen LLM provider.
6. The response comes back, placeholders are swapped for originals, and your app gets a normal-looking reply.

---

## Why It Exists

*   **Context-aware anonymization:** A local LLM understands that "Goldman Sachs" is PII but "the Purchaser" is a legal term — something regex and NER can't do reliably.
*   **Semantic placeholders:** `INVESTMENT_BANK_01`, not `ORGANIZATION_01`. The downstream LLM keeps enough context to reason well.
*   **Safety net:** Presidio runs after the LLM as a second pass. Configurable as `warn` (auto-fix), `strict` (reject), or `off`.
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
| `ANONYMIZATION_MODE` | `llm` | `llm` (context-aware) or `mechanical` (NER-only) |
| `SAFETY_NET_STRICTNESS` | `warn` | `warn` (auto-fix), `strict` (reject), or `off` |

When running through Docker Compose, the service automatically uses `host.docker.internal` so the container can reach Ollama on your Mac.

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

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
