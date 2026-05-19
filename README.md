<p align="left">
  <img src="docs/assets/jude-logo.png" alt="Hey Jude logo" width="120">
</p>

# Hey Jude

**Risk-mitigating pseudonymization gateway for legal LLM workflows.**

<p align="left">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="Docker Compose" src="https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white">
  <img alt="OpenAI-compatible" src="https://img.shields.io/badge/OpenAI--compatible-API-111827?style=flat-square">
  <img alt="License AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-111827?style=flat-square">
</p>

Hey Jude is an open-source mitigation proxy designed to reduce sensitive data exposure when using LLMs in legal workflows. It detects and pseudonymizes common PII, organization names, and selected sensitive terms before a prompt is routed onward, then attempts to restore the original details in the response.

This tool is a helper layer for data minimization. It does not guarantee complete detection of all sensitive information and should be used as part of a broader confidentiality strategy.

---

## Why It Exists

*   **Smart pseudonymization:** Replaces people, organizations, email addresses, and phone numbers with safer synthetic values.
*   **Local anonymization model:** Uses a local OpenAI-compatible endpoint, usually Ollama, before anything is routed onward.
*   **Drop-in API shape:** Exposes OpenAI, Anthropic, and Gemini-compatible endpoints so existing SDK clients can point at the gateway.
*   **Default local demo:** Runs without a cloud LLM key by using Ollama as both the anonymization model and the demo destination.
*   **External routing when ready:** Advanced users can route anonymized prompts to OpenAI, Anthropic, Gemini, Azure, or any LiteLLM-compatible provider.

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
python3 scripts/test_gateway.py
```

The default setup is fully local: Redis runs in Docker, and Ollama runs on your host machine.

---

## Default Configuration

The checked-in defaults are intended to work for most new users without model or provider selection.

| Variable | Default | Purpose |
|----------|---------|---------|
| `REDIS_URL` | `redis://localhost:6379/0` | Temporary mapping memory |
| `API_KEY` | `sk-heyjude-dev` | Gateway authentication |
| `LOCAL_LLM_URL` | `http://localhost:11434/v1` | Local anonymization endpoint |
| `LOCAL_LLM_MODEL` | `qwen3.5:4b` | Local anonymization model |
| `EXTERNAL_LLM_MODEL` | `ollama_chat/qwen3.5:4b` | Demo destination model via LiteLLM |
| `EXTERNAL_LLM_API_BASE` | `http://localhost:11434` | LiteLLM API base for Ollama |

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
