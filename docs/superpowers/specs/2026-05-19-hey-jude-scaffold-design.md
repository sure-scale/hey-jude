# Hey Jude: Context-Preserving Pseudonymization Gateway

## Overview

Hey Jude is an open-source pseudonymization gateway for law firms. It intercepts LLM prompts, uses a local LLM to replace sensitive entities with contextually accurate fictional entities, routes the sanitized prompt to an external enterprise LLM (e.g., Azure AI Foundry), and reverses the entities in the response before returning it to the user.

## Core Technologies

- **Language:** Python 3.11+
- **Web Framework:** FastAPI (async)
- **LLM Gateway:** LiteLLM (external model routing, standardized API calls)
- **Entity Detection:** Microsoft Presidio (PII/entity flagging)
- **State Management:** Redis (ephemeral mapping storage with TTL)
- **Local Inference:** Any OpenAI-compatible endpoint (Ollama, LM Studio, etc.)
- **Packaging:** Docker Compose (gateway + Redis containers)

## Architecture

Monolithic FastAPI application with service modules. Single process, single entry point. Services communicate via function calls — no message queues or microservices.

### Project Structure

```
hey-jude/
├── src/
│   └── hey_jude/
│       ├── __init__.py
│       ├── main.py              # FastAPI app factory, lifespan (Redis init)
│       ├── config.py            # Pydantic BaseSettings from env vars
│       ├── routes.py            # /v1/chat/completions endpoint
│       ├── models.py            # Request/response Pydantic models (OpenAI schema)
│       ├── redis_client.py      # Redis connection + get/set mapping helpers
│       └── services/
│           ├── __init__.py
│           ├── detector.py      # Presidio AnalyzerEngine wrapper
│           ├── substitutor.py   # Local LLM caller (analyze, decide, execute)
│           ├── router.py        # LiteLLM acompletion call to external model
│           └── mapper.py        # Reverse entity substitution in responses
├── tests/
│   ├── conftest.py
│   ├── test_routes.py
│   ├── test_detector.py
│   ├── test_substitutor.py
│   ├── test_router.py
│   └── test_mapper.py
├── docker-compose.yml           # Gateway + Redis
├── Dockerfile
├── pyproject.toml               # Python 3.11+, dependencies
├── .env.example                 # Template for required env vars
├── .gitignore
├── LICENSE                      # MIT
└── README.md
```

## Configuration

All configuration via environment variables, loaded through Pydantic `BaseSettings` in `config.py`.

### Required Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `REDIS_TTL_SECONDS` | Mapping expiry in seconds | `3600` |
| `LOCAL_LLM_URL` | OpenAI-compatible local endpoint | `http://localhost:11434/v1` |
| `LOCAL_LLM_MODEL` | Model name for local inference | `llama3.2` |
| `EXTERNAL_LLM_MODEL` | LiteLLM model identifier | `azure/gpt-4o` |
| `API_KEY` | Gateway authentication key | `sk-heyjude-...` |

### Entity Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `PRESIDIO_ENTITIES` | Which entity types to detect | `["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER"]` |
| `ENTITY_STRATEGIES` | Per-type replacement strategy | `{"PERSON": "llm", "ORGANIZATION": "llm", "EMAIL_ADDRESS": "deterministic", "PHONE_NUMBER": "deterministic"}` |

### Guardrail Toggles

| Toggle | Default | Behavior |
|--------|---------|----------|
| `ALWAYS_FULL_ANONYMIZATION` | `false` | Skip sensitivity analysis, always treat as high sensitivity |
| `ANONYMIZE_PRODUCT_NAMES` | `true` | Replace product/service names alongside organizations |
| `ABSTRACT_RELATIONSHIPS` | `true` | When high sensitivity, rephrase structural patterns in addition to entity replacement |
| `PASSTHROUGH_SYSTEM_MESSAGES` | `false` | Whether system messages bypass anonymization |
| `MAX_CONTEXT_WINDOW` | `500` | Max chars of surrounding context sent to local LLM for analysis |
| `ALLOW_CLARIFICATION_REQUESTS` | `true` | Allow gateway to ask user for clarification when unsure |

## API Design

### Endpoint: `POST /v1/chat/completions`

Mimics the OpenAI Chat Completions API schema. Non-streaming only for initial release.

### Authentication

`X-API-Key` header or `Authorization: Bearer <key>` header. Key validated against `API_KEY` env var.

### Request Model

```python
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
```

### Response Model

Standard OpenAI chat completion response with an additional `heyjude_metadata` field:

```python
class HeyJudeMetadata(BaseModel):
    request_id: str
    entities_detected: int
    sensitivity: str  # "low", "high", or "none"
    status: str  # "completed" or "clarification_needed"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    heyjude_metadata: HeyJudeMetadata | None = None
```

### Health Check: `GET /health`

Returns gateway status including Redis connectivity and local LLM reachability.

## Core Workflow

### Data Flow (routes.py orchestration)

```
Client Request
    |
    v
[1] routes.py: validate request, check API key, extract messages
    |
    v
[2] detector.py: run Presidio on each user/assistant message
    |  returns: list[DetectedEntity] with text, type, start, end, score
    |
    v
[3] substitutor.py: three-phase anonymization
    |  Phase 1 - ANALYZE: classify sensitivity, generate context descriptors
    |  Phase 2 - DECIDE: choose anonymization strategy based on sensitivity + config
    |  Phase 3 - EXECUTE: generate fictional replacements + sanitized text
    |  returns: SubstitutionResult (mapping_dict, sanitized_messages, context_descriptors, sensitivity)
    |
    v
[4] redis_client.py: store mapping_dict + context with request_id, enforce REDIS_TTL_SECONDS
    |
    v
[5] router.py: inject context preamble, call litellm.acompletion() with sanitized messages
    |  returns: external LLM response containing synthetic entities
    |
    v
[6] mapper.py: retrieve mapping from Redis, reverse-replace all synthetic -> original entities
    |
    v
[7] routes.py: return final de-anonymized response to client
```

### No Entities Path

If detector finds no entities, skip steps 3-4 and 6. Route directly to external LLM via step 5.

### Clarification Path

If local LLM determines it needs clarification (and `ALLOW_CLARIFICATION_REQUESTS` is true):
1. Original request stored in Redis with request_id
2. Gateway returns standard chat completion response where assistant message is the clarification question
3. `heyjude_metadata.status` set to `"clarification_needed"`
4. Client resends with clarification, referencing request_id
5. Gateway retrieves original context and proceeds with anonymization

## Service Details

### detector.py

Wraps Microsoft Presidio's `AnalyzerEngine`. Configurable entity types via `PRESIDIO_ENTITIES`.

```python
@dataclass
class DetectedEntity:
    text: str
    entity_type: str
    start: int
    end: int
    score: float

async def detect_entities(text: str, settings: Settings) -> list[DetectedEntity]:
    """Run Presidio analysis on text, return detected entities."""
```

### substitutor.py

Three-phase anonymization engine using local LLM.

**Phase 1 — Analyze:** Send entities + query to local LLM. Classify sensitivity level (`low` or `high`). Generate context descriptors for each entity (e.g., `"Apple" -> "large technology company"`).

**Phase 2 — Decide:** Based on sensitivity level and config guardrails, determine:
- Which entities get LLM-generated fictional replacements
- Which get deterministic pattern replacements
- Whether structural patterns need abstraction (high sensitivity + `ABSTRACT_RELATIONSHIPS=true`)
- Whether to request user clarification

**Phase 3 — Execute:** Generate replacements and perform substitution.

Prompt to local LLM:

```xml
<task>
Analyze this legal professional's query for sensitive entity handling.
</task>

<entities>
[{"text": "Meta", "type": "ORGANIZATION"}, {"text": "Microsoft", "type": "ORGANIZATION"}, {"text": "Instagram", "type": "ORGANIZATION"}]
</entities>

<query>
Bring up all Meta vs Microsoft cases that involve IP around Instagram
</query>

<instructions>
1. Classify sensitivity: "low" if entity replacement alone prevents identification, "high" if structural patterns or relationships could de-anonymize.
2. For each entity, provide a brief context descriptor (what kind of entity it is, without identifying it).
3. Generate fictional replacement names that preserve the entity's role and domain. Always use fictional names, never descriptive phrases.
4. If high sensitivity and structural abstraction is enabled: also rephrase the query to obscure identifying structural patterns while preserving the legal question's intent.
5. If you are unsure about the appropriate anonymization strategy, set needs_clarification to true and provide a clarification question.
6. Return JSON with: sensitivity, reasoning, mapping, context_descriptors, sanitized_text, needs_clarification, clarification_question
</instructions>
```

Expected local LLM response:

```json
{
  "sensitivity": "high",
  "reasoning": "Cross-referencing two named companies in a specific IP dispute with a named product could identify the real case",
  "mapping": {
    "Meta": "Vertex Holdings",
    "Microsoft": "Pinnacle Systems",
    "Instagram": "Photogram"
  },
  "context_descriptors": {
    "Vertex Holdings": "large social media conglomerate",
    "Pinnacle Systems": "major technology corporation",
    "Photogram": "social media platform subsidiary of Vertex Holdings"
  },
  "sanitized_text": "Bring up all Vertex Holdings vs Pinnacle Systems cases that involve IP around Photogram",
  "needs_clarification": false,
  "clarification_question": null
}
```

**Deterministic replacements** for non-LLM entity types:
- `EMAIL_ADDRESS` -> `user_N@example.com`
- `PHONE_NUMBER` -> `555-0100+N`
- Counter N increments per request to avoid collisions

```python
@dataclass
class SubstitutionResult:
    mapping: dict[str, str]           # original -> synthetic
    reverse_mapping: dict[str, str]   # synthetic -> original
    context_descriptors: dict[str, str]
    sanitized_messages: list[ChatMessage]
    sensitivity: str
    needs_clarification: bool
    clarification_question: str | None

async def substitute_entities(
    messages: list[ChatMessage],
    entities: list[DetectedEntity],
    settings: Settings,
) -> SubstitutionResult:
    """Three-phase anonymization: analyze, decide, execute."""
```

### router.py

Wraps LiteLLM's `acompletion` call. Injects context preamble with entity descriptors into the system message:

```
[Context: For this query, the following entities are referenced:
- Vertex Holdings: large social media conglomerate
- Pinnacle Systems: major technology corporation
- Photogram: social media platform subsidiary of Vertex Holdings]
```

```python
async def route_completion(
    messages: list[ChatMessage],
    context_descriptors: dict[str, str],
    model: str,
    **kwargs,
) -> dict:
    """Send sanitized messages to external LLM via LiteLLM."""
```

### mapper.py

Reverse substitution: replace all synthetic entities in the response with originals.

Must handle:
- Case-insensitive matching (LLM might change casing)
- Longest-match-first replacement (avoid partial replacements, e.g., "Vertex" inside "Vertex Holdings")
- Multiple occurrences across the response

```python
async def reverse_map(
    response_text: str,
    request_id: str,
    redis_client: RedisClient,
) -> str:
    """Retrieve mapping from Redis and reverse all synthetic entities."""
```

### redis_client.py

Thin wrapper around `redis.asyncio`. Stores and retrieves mapping dictionaries as JSON.

```python
class RedisClient:
    async def store_mapping(self, request_id: str, mapping: dict, ttl: int) -> None:
        """Store mapping dict with TTL enforcement."""

    async def get_mapping(self, request_id: str) -> dict | None:
        """Retrieve mapping dict. Returns None if expired/missing."""

    async def store_request(self, request_id: str, request_data: dict, ttl: int) -> None:
        """Store original request for clarification flow."""

    async def get_request(self, request_id: str) -> dict | None:
        """Retrieve stored request for clarification follow-up."""

    async def health_check(self) -> bool:
        """Ping Redis to verify connectivity."""
```

## Error Handling

**Core invariant:** If anonymization cannot complete successfully, the gateway MUST NOT forward the original text to the external LLM. Fail closed, never fail open.

| Failure | Response | Rationale |
|---------|----------|-----------|
| Redis unreachable | 503 Service Unavailable | Cannot store mapping = cannot de-anonymize response |
| Local LLM unreachable | 503 Service Unavailable | Cannot anonymize = must not send original text externally |
| Local LLM returns invalid JSON | Retry once with stricter prompt, then 502 Bad Gateway | Local LLMs can be unreliable with structured output |
| External LLM fails | Pass through upstream error code from LiteLLM | Standard gateway behavior |
| No entities detected | Skip anonymization, route directly | Nothing to protect |
| Redis mapping expired before response arrives | 504 Gateway Timeout | Cannot reverse entities without mapping |
| Invalid or missing API key | 401 Unauthorized | Standard auth |
| Malformed request body | 422 Unprocessable Entity (FastAPI default) | Standard validation |

## Testing Strategy

Unit tests with dependency injection. Tests requiring Redis use a real Redis instance via docker-compose test profile.

| Test file | Scope |
|-----------|-------|
| `test_detector.py` | Presidio identifies known entity types in sample legal text |
| `test_substitutor.py` | Given injected local LLM responses, verify mapping dict, sanitized text, sensitivity classification. Test both low/high sensitivity paths. Test clarification path. |
| `test_mapper.py` | Reverse substitution handles overlapping entities, partial matches, case sensitivity, longest-match-first |
| `test_router.py` | Verify LiteLLM called with sanitized (not original) text, context preamble injected correctly |
| `test_routes.py` | End-to-end flow with injected dependencies. Auth check, full pipeline, error responses, clarification flow. |

## Docker Compose

```yaml
services:
  gateway:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

Local LLM (Ollama/LM Studio) assumed running on host, accessible via `LOCAL_LLM_URL`.

## Scope Boundaries (Initial Scaffold)

### In scope
- Non-streaming `/v1/chat/completions` endpoint
- API key authentication
- Presidio entity detection with configurable types
- Three-phase substitutor with local LLM (analyze, decide, execute)
- Configurable per-type replacement strategy (LLM vs deterministic)
- Guardrail config toggles
- Clarification flow
- Redis ephemeral mapping storage with TTL
- LiteLLM routing to external model
- Reverse entity mapping
- Docker Compose deployment
- Unit tests

### Out of scope (future phases)
- Streaming (SSE) support
- Persistent audit logging
- OAuth2/OIDC authentication
- Multi-user / multi-tenant support
- Web UI
- Conversation history / multi-turn context tracking
- Custom Presidio recognizers
- Kubernetes / Helm deployment

## Verification

1. `docker compose up` starts gateway + Redis
2. Verify `GET /health` returns OK with Redis and local LLM connectivity status
3. Send test request to `POST /v1/chat/completions` with known entities
4. Verify response contains original (de-anonymized) entities, not synthetic ones
5. Verify Redis mapping was stored and expired after TTL
6. Verify sending request without API key returns 401
7. Run `pytest` with Redis running to execute full test suite
