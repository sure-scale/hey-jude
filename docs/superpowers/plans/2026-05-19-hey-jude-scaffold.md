# Hey Jude Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the Hey Jude pseudonymization gateway — a FastAPI app that detects entities in LLM prompts, replaces them with context-aware fictional equivalents via a local LLM, routes sanitized prompts to an external enterprise LLM, and reverses substitutions in the response.

**Architecture:** Monolithic FastAPI app with service modules (`detector`, `substitutor`, `router`, `mapper`) orchestrated by a single `/v1/chat/completions` endpoint. Redis stores ephemeral entity mappings. Local LLM (any OpenAI-compatible endpoint) performs intelligent anonymization. LiteLLM handles external model routing.

**Tech Stack:** Python 3.11+, FastAPI, LiteLLM, Microsoft Presidio, Redis (async), httpx, Pydantic v2, Docker Compose

**Spec:** `docs/superpowers/specs/2026-05-19-hey-jude-scaffold-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, pytest config |
| `src/hey_jude/__init__.py` | Package marker, version |
| `src/hey_jude/config.py` | Pydantic BaseSettings — all env vars, guardrail toggles |
| `src/hey_jude/models.py` | Pydantic models: ChatMessage, ChatCompletionRequest/Response, HeyJudeMetadata, DetectedEntity, SubstitutionResult |
| `src/hey_jude/redis_client.py` | RedisClient class — store/get mappings and requests, health check |
| `src/hey_jude/services/__init__.py` | Package marker |
| `src/hey_jude/services/detector.py` | Presidio AnalyzerEngine wrapper — detect_entities() |
| `src/hey_jude/services/substitutor.py` | Three-phase anonymization engine — call local LLM, deterministic fallbacks |
| `src/hey_jude/services/router.py` | LiteLLM acompletion wrapper — inject context preamble, route to external model |
| `src/hey_jude/services/mapper.py` | Reverse entity substitution — case-insensitive, longest-match-first |
| `src/hey_jude/routes.py` | FastAPI router — /v1/chat/completions, /health, auth middleware |
| `src/hey_jude/main.py` | App factory, lifespan (Redis init/teardown), include router |
| `tests/conftest.py` | Shared fixtures: settings, redis client, FastAPI test client |
| `tests/test_detector.py` | Presidio entity detection tests |
| `tests/test_substitutor.py` | Substitutor tests — low/high sensitivity, deterministic, clarification |
| `tests/test_mapper.py` | Reverse mapping tests — case insensitive, longest match, overlaps |
| `tests/test_router.py` | LiteLLM routing tests — preamble injection, passthrough params |
| `tests/test_routes.py` | Integration tests — full pipeline, auth, errors, clarification flow |
| `Dockerfile` | Multi-stage Python image |
| `docker-compose.yml` | Gateway + Redis services |
| `.env.example` | Env var template |
| `.gitignore` | Python/Docker ignores |
| `LICENSE` | MIT license |
| `README.md` | Project overview, quickstart, config reference |

---

## Task 1: Project Skeleton and Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/hey_jude/__init__.py`
- Create: `src/hey_jude/config.py`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `LICENSE`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "hey-jude"
version = "0.1.0"
description = "Context-preserving pseudonymization gateway for law firms"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "litellm>=1.40.0",
    "presidio-analyzer>=2.2.0",
    "presidio-anonymizer>=2.2.0",
    "redis[hiredis]>=5.0.0",
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-httpx>=0.30.0",
    "httpx>=0.27.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.hatch.build.targets.wheel]
packages = ["src/hey_jude"]
```

- [ ] **Step 2: Create package init**

`src/hey_jude/__init__.py`:
```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Create config.py**

`src/hey_jude/config.py`:
```python
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 3600

    local_llm_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "llama3.2"

    external_llm_model: str = "azure/gpt-4o"

    api_key: str = "sk-heyjude-dev"

    presidio_entities: list[str] = Field(
        default=["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER"]
    )
    entity_strategies: dict[str, str] = Field(
        default={
            "PERSON": "llm",
            "ORGANIZATION": "llm",
            "EMAIL_ADDRESS": "deterministic",
            "PHONE_NUMBER": "deterministic",
        }
    )

    always_full_anonymization: bool = False
    anonymize_product_names: bool = True
    abstract_relationships: bool = True
    passthrough_system_messages: bool = False
    max_context_window: int = 500
    allow_clarification_requests: bool = True


settings = Settings()
```

- [ ] **Step 4: Create .env.example**

`.env.example`:
```bash
# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_TTL_SECONDS=3600

# Local LLM (OpenAI-compatible endpoint)
LOCAL_LLM_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=llama3.2

# External LLM (via LiteLLM)
EXTERNAL_LLM_MODEL=azure/gpt-4o

# Gateway auth
API_KEY=sk-heyjude-change-me

# Entity detection
PRESIDIO_ENTITIES=["PERSON","ORGANIZATION","EMAIL_ADDRESS","PHONE_NUMBER"]
ENTITY_STRATEGIES={"PERSON":"llm","ORGANIZATION":"llm","EMAIL_ADDRESS":"deterministic","PHONE_NUMBER":"deterministic"}

# Guardrails
ALWAYS_FULL_ANONYMIZATION=false
ANONYMIZE_PRODUCT_NAMES=true
ABSTRACT_RELATIONSHIPS=true
PASSTHROUGH_SYSTEM_MESSAGES=false
MAX_CONTEXT_WINDOW=500
ALLOW_CLARIFICATION_REQUESTS=true
```

- [ ] **Step 5: Create .gitignore**

`.gitignore`:
```
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.eggs/
*.egg
.env
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.log
```

- [ ] **Step 6: Create LICENSE**

`LICENSE`: Standard MIT license text with `2026 Hey Jude Contributors`.

- [ ] **Step 7: Install project in dev mode and verify**

Run: `pip install -e ".[dev]"`
Expected: Installs successfully, `python -c "from hey_jude.config import Settings; print(Settings())"` prints defaults.

- [ ] **Step 8: Commit**

```bash
git init
git add pyproject.toml src/hey_jude/__init__.py src/hey_jude/config.py .env.example .gitignore LICENSE
git commit -m "feat: project skeleton with config and dependencies"
```

---

## Task 2: Pydantic Models

**Files:**
- Create: `src/hey_jude/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing test for request/response models**

`tests/test_models.py`:
```python
from hey_jude.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DetectedEntity,
    HeyJudeMetadata,
    SubstitutionResult,
    Usage,
)


def test_chat_message_valid():
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_chat_completion_request_minimal():
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role="user", content="Hi")],
    )
    assert req.model == "gpt-4o"
    assert len(req.messages) == 1
    assert req.temperature is None


def test_chat_completion_request_full():
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role="user", content="Hi")],
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )
    assert req.temperature == 0.7
    assert req.max_tokens == 100


def test_chat_completion_response():
    resp = ChatCompletionResponse(
        id="chatcmpl-123",
        created=1700000000,
        model="gpt-4o",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content="Hello back"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        heyjude_metadata=HeyJudeMetadata(
            request_id="req-abc",
            entities_detected=2,
            sensitivity="high",
            status="completed",
        ),
    )
    assert resp.id == "chatcmpl-123"
    assert resp.choices[0].message.content == "Hello back"
    assert resp.heyjude_metadata.sensitivity == "high"


def test_detected_entity():
    entity = DetectedEntity(
        text="Microsoft",
        entity_type="ORGANIZATION",
        start=10,
        end=19,
        score=0.95,
    )
    assert entity.text == "Microsoft"
    assert entity.entity_type == "ORGANIZATION"


def test_substitution_result():
    result = SubstitutionResult(
        mapping={"Microsoft": "Pinnacle Systems"},
        reverse_mapping={"Pinnacle Systems": "Microsoft"},
        context_descriptors={"Pinnacle Systems": "major technology corporation"},
        sanitized_messages=[ChatMessage(role="user", content="Tell me about Pinnacle Systems")],
        sensitivity="low",
        needs_clarification=False,
        clarification_question=None,
    )
    assert result.mapping["Microsoft"] == "Pinnacle Systems"
    assert result.reverse_mapping["Pinnacle Systems"] == "Microsoft"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -v`
Expected: ImportError — `hey_jude.models` does not exist.

- [ ] **Step 3: Implement models.py**

`src/hey_jude/models.py`:
```python
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel


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


class ChoiceMessage(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class HeyJudeMetadata(BaseModel):
    request_id: str
    entities_detected: int
    sensitivity: str
    status: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    heyjude_metadata: HeyJudeMetadata | None = None


@dataclass
class DetectedEntity:
    text: str
    entity_type: str
    start: int
    end: int
    score: float


@dataclass
class SubstitutionResult:
    mapping: dict[str, str]
    reverse_mapping: dict[str, str]
    context_descriptors: dict[str, str]
    sanitized_messages: list[ChatMessage]
    sensitivity: str
    needs_clarification: bool
    clarification_question: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hey_jude/models.py tests/test_models.py
git commit -m "feat: pydantic models for OpenAI-compatible request/response schema"
```

---

## Task 3: Redis Client

**Files:**
- Create: `src/hey_jude/redis_client.py`
- Create: `tests/conftest.py`
- Create: `tests/test_redis_client.py`

Requires running Redis on localhost:6379 (use `docker run -d -p 6379:6379 redis:7-alpine` or docker-compose).

- [ ] **Step 1: Write failing tests**

`tests/conftest.py`:
```python
import pytest
from hey_jude.config import Settings
from hey_jude.redis_client import RedisClient


@pytest.fixture
def test_settings():
    return Settings(
        redis_url="redis://localhost:6379/1",
        redis_ttl_seconds=10,
        api_key="sk-test-key",
    )


@pytest.fixture
async def redis_client(test_settings):
    client = RedisClient(test_settings.redis_url)
    await client.connect()
    await client._redis.flushdb()
    yield client
    await client._redis.flushdb()
    await client.close()
```

`tests/test_redis_client.py`:
```python
import json

import pytest


async def test_store_and_get_mapping(redis_client):
    mapping = {"Microsoft": "Pinnacle Systems", "John": "James"}
    await redis_client.store_mapping("req-1", mapping, ttl=60)
    result = await redis_client.get_mapping("req-1")
    assert result == mapping


async def test_get_mapping_missing_key(redis_client):
    result = await redis_client.get_mapping("nonexistent")
    assert result is None


async def test_mapping_ttl_enforced(redis_client):
    await redis_client.store_mapping("req-2", {"a": "b"}, ttl=1)
    import asyncio
    await asyncio.sleep(1.1)
    result = await redis_client.get_mapping("req-2")
    assert result is None


async def test_store_and_get_request(redis_client):
    request_data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    await redis_client.store_request("req-3", request_data, ttl=60)
    result = await redis_client.get_request("req-3")
    assert result == request_data


async def test_get_request_missing_key(redis_client):
    result = await redis_client.get_request("nonexistent")
    assert result is None


async def test_health_check(redis_client):
    assert await redis_client.health_check() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_redis_client.py -v`
Expected: ImportError — `hey_jude.redis_client` has no `RedisClient`.

- [ ] **Step 3: Implement redis_client.py**

`src/hey_jude/redis_client.py`:
```python
import json

import redis.asyncio as aioredis


class RedisClient:
    def __init__(self, url: str):
        self._url = url
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._redis = aioredis.from_url(self._url, decode_responses=True)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    async def store_mapping(self, request_id: str, mapping: dict, ttl: int) -> None:
        key = f"heyjude:mapping:{request_id}"
        await self._redis.set(key, json.dumps(mapping), ex=ttl)

    async def get_mapping(self, request_id: str) -> dict | None:
        key = f"heyjude:mapping:{request_id}"
        data = await self._redis.get(key)
        if data is None:
            return None
        return json.loads(data)

    async def store_request(self, request_id: str, request_data: dict, ttl: int) -> None:
        key = f"heyjude:request:{request_id}"
        await self._redis.set(key, json.dumps(request_data), ex=ttl)

    async def get_request(self, request_id: str) -> dict | None:
        key = f"heyjude:request:{request_id}"
        data = await self._redis.get(key)
        if data is None:
            return None
        return json.loads(data)

    async def health_check(self) -> bool:
        try:
            return await self._redis.ping()
        except Exception:
            return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_redis_client.py -v`
Expected: All 6 tests PASS (requires Redis running on localhost:6379).

- [ ] **Step 5: Commit**

```bash
git add src/hey_jude/redis_client.py tests/conftest.py tests/test_redis_client.py
git commit -m "feat: async Redis client for mapping and request storage"
```

---

## Task 4: Entity Detector (Presidio)

**Files:**
- Create: `src/hey_jude/services/__init__.py`
- Create: `src/hey_jude/services/detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: Write failing tests**

`tests/test_detector.py`:
```python
import pytest
from hey_jude.config import Settings
from hey_jude.models import DetectedEntity
from hey_jude.services.detector import detect_entities


@pytest.fixture
def detector_settings():
    return Settings(
        presidio_entities=["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    )


async def test_detect_person_and_org(detector_settings):
    text = "John Smith works at Microsoft on the Azure platform."
    entities = await detect_entities(text, detector_settings)
    entity_types = {e.entity_type for e in entities}
    entity_texts = {e.text for e in entities}
    assert "PERSON" in entity_types
    assert "ORGANIZATION" in entity_types
    assert "John Smith" in entity_texts


async def test_detect_email(detector_settings):
    text = "Contact us at john@example.com for details."
    entities = await detect_entities(text, detector_settings)
    email_entities = [e for e in entities if e.entity_type == "EMAIL_ADDRESS"]
    assert len(email_entities) >= 1
    assert email_entities[0].text == "john@example.com"


async def test_detect_no_entities(detector_settings):
    text = "The weather is nice today."
    entities = await detect_entities(text, detector_settings)
    assert entities == []


async def test_detected_entity_has_position(detector_settings):
    text = "John Smith is a lawyer."
    entities = await detect_entities(text, detector_settings)
    person_entities = [e for e in entities if e.entity_type == "PERSON"]
    assert len(person_entities) >= 1
    entity = person_entities[0]
    assert entity.start >= 0
    assert entity.end > entity.start
    assert text[entity.start:entity.end] == entity.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detector.py -v`
Expected: ImportError — `hey_jude.services.detector` does not exist.

- [ ] **Step 3: Create services package**

`src/hey_jude/services/__init__.py`:
```python
```

- [ ] **Step 4: Implement detector.py**

`src/hey_jude/services/detector.py`:
```python
from presidio_analyzer import AnalyzerEngine

from hey_jude.config import Settings
from hey_jude.models import DetectedEntity

_engine: AnalyzerEngine | None = None


def _get_engine() -> AnalyzerEngine:
    global _engine
    if _engine is None:
        _engine = AnalyzerEngine()
    return _engine


async def detect_entities(text: str, settings: Settings) -> list[DetectedEntity]:
    engine = _get_engine()
    results = engine.analyze(
        text=text,
        entities=settings.presidio_entities,
        language="en",
    )
    entities = []
    for result in results:
        entities.append(
            DetectedEntity(
                text=text[result.start:result.end],
                entity_type=result.entity_type,
                start=result.start,
                end=result.end,
                score=result.score,
            )
        )
    return entities
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_detector.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hey_jude/services/__init__.py src/hey_jude/services/detector.py tests/test_detector.py
git commit -m "feat: Presidio entity detector with configurable entity types"
```

---

## Task 5: Entity Mapper (Reverse Substitution)

**Files:**
- Create: `src/hey_jude/services/mapper.py`
- Create: `tests/test_mapper.py`

Building mapper before substitutor because it has no external dependencies and tests are self-contained.

- [ ] **Step 1: Write failing tests**

`tests/test_mapper.py`:
```python
import pytest
from hey_jude.services.mapper import reverse_map_text


def test_simple_replacement():
    text = "Pinnacle Systems released a new product."
    reverse_mapping = {"Pinnacle Systems": "Microsoft"}
    result = reverse_map_text(text, reverse_mapping)
    assert result == "Microsoft released a new product."


def test_multiple_replacements():
    text = "Vertex Holdings sued Pinnacle Systems over Photogram."
    reverse_mapping = {
        "Vertex Holdings": "Meta",
        "Pinnacle Systems": "Microsoft",
        "Photogram": "Instagram",
    }
    result = reverse_map_text(text, reverse_mapping)
    assert result == "Meta sued Microsoft over Instagram."


def test_case_insensitive_matching():
    text = "pinnacle systems announced earnings."
    reverse_mapping = {"Pinnacle Systems": "Microsoft"}
    result = reverse_map_text(text, reverse_mapping)
    assert result == "Microsoft announced earnings."


def test_longest_match_first():
    text = "Vertex Holdings International is a big company. Vertex Holdings is too."
    reverse_mapping = {
        "Vertex Holdings International": "Meta Platforms Inc",
        "Vertex Holdings": "Meta",
    }
    result = reverse_map_text(text, reverse_mapping)
    assert result == "Meta Platforms Inc is a big company. Meta is too."


def test_multiple_occurrences():
    text = "Pinnacle Systems vs Pinnacle Systems in court."
    reverse_mapping = {"Pinnacle Systems": "Microsoft"}
    result = reverse_map_text(text, reverse_mapping)
    assert result == "Microsoft vs Microsoft in court."


def test_no_mapping_passthrough():
    text = "No entities here."
    result = reverse_map_text(text, {})
    assert result == "No entities here."


def test_no_match_passthrough():
    text = "Some other company did a thing."
    reverse_mapping = {"Pinnacle Systems": "Microsoft"}
    result = reverse_map_text(text, reverse_mapping)
    assert result == "Some other company did a thing."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mapper.py -v`
Expected: ImportError — `hey_jude.services.mapper` does not exist.

- [ ] **Step 3: Implement mapper.py**

`src/hey_jude/services/mapper.py`:
```python
import re

from hey_jude.redis_client import RedisClient


def reverse_map_text(text: str, reverse_mapping: dict[str, str]) -> str:
    if not reverse_mapping:
        return text
    sorted_keys = sorted(reverse_mapping.keys(), key=len, reverse=True)
    for synthetic in sorted_keys:
        original = reverse_mapping[synthetic]
        pattern = re.compile(re.escape(synthetic), re.IGNORECASE)
        text = pattern.sub(original, text)
    return text


async def reverse_map(
    response_text: str,
    request_id: str,
    redis_client: RedisClient,
) -> str:
    mapping = await redis_client.get_mapping(request_id)
    if mapping is None:
        raise LookupError(f"Mapping expired or missing for request {request_id}")
    reverse_mapping = {v: k for k, v in mapping.items()}
    return reverse_map_text(response_text, reverse_mapping)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mapper.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hey_jude/services/mapper.py tests/test_mapper.py
git commit -m "feat: reverse entity mapper with case-insensitive longest-match-first"
```

---

## Task 6: Substitutor (Three-Phase Anonymization Engine)

**Files:**
- Create: `src/hey_jude/services/substitutor.py`
- Create: `tests/test_substitutor.py`

This is the most complex service. Uses httpx to call the local LLM's OpenAI-compatible endpoint.

- [ ] **Step 1: Write failing tests**

`tests/test_substitutor.py`:
```python
import json

import pytest
from hey_jude.config import Settings
from hey_jude.models import ChatMessage, DetectedEntity
from hey_jude.services.substitutor import (
    _build_deterministic_replacements,
    _build_prompt,
    _parse_llm_response,
    substitute_entities,
)


def test_build_deterministic_replacements():
    entities = [
        DetectedEntity(text="john@acme.com", entity_type="EMAIL_ADDRESS", start=0, end=13, score=0.9),
        DetectedEntity(text="555-1234", entity_type="PHONE_NUMBER", start=20, end=28, score=0.9),
    ]
    strategies = {"EMAIL_ADDRESS": "deterministic", "PHONE_NUMBER": "deterministic"}
    mapping = _build_deterministic_replacements(entities, strategies)
    assert mapping["john@acme.com"] == "user_1@example.com"
    assert mapping["555-1234"] == "555-0101"


def test_build_deterministic_skips_llm_entities():
    entities = [
        DetectedEntity(text="Microsoft", entity_type="ORGANIZATION", start=0, end=9, score=0.9),
        DetectedEntity(text="john@acme.com", entity_type="EMAIL_ADDRESS", start=15, end=28, score=0.9),
    ]
    strategies = {"ORGANIZATION": "llm", "EMAIL_ADDRESS": "deterministic"}
    mapping = _build_deterministic_replacements(entities, strategies)
    assert "Microsoft" not in mapping
    assert "john@acme.com" in mapping


def test_build_prompt():
    entities = [
        DetectedEntity(text="Meta", entity_type="ORGANIZATION", start=0, end=4, score=0.9),
        DetectedEntity(text="Microsoft", entity_type="ORGANIZATION", start=8, end=17, score=0.9),
    ]
    query = "Meta vs Microsoft IP case"
    prompt = _build_prompt(entities, query)
    assert "Meta" in prompt
    assert "Microsoft" in prompt
    assert "ORGANIZATION" in prompt
    assert "sensitivity" in prompt


def test_parse_llm_response_valid():
    raw = json.dumps({
        "sensitivity": "high",
        "reasoning": "test",
        "mapping": {"Meta": "Vertex Holdings"},
        "context_descriptors": {"Vertex Holdings": "social media company"},
        "sanitized_text": "Vertex Holdings did a thing",
        "needs_clarification": False,
        "clarification_question": None,
    })
    result = _parse_llm_response(raw)
    assert result["sensitivity"] == "high"
    assert result["mapping"]["Meta"] == "Vertex Holdings"


def test_parse_llm_response_invalid():
    with pytest.raises(ValueError, match="Failed to parse"):
        _parse_llm_response("this is not json at all")


def test_parse_llm_response_json_in_markdown():
    raw = '```json\n{"sensitivity": "low", "reasoning": "", "mapping": {}, "context_descriptors": {}, "sanitized_text": "hello", "needs_clarification": false, "clarification_question": null}\n```'
    result = _parse_llm_response(raw)
    assert result["sensitivity"] == "low"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_substitutor.py -v`
Expected: ImportError — `hey_jude.services.substitutor` does not exist.

- [ ] **Step 3: Implement substitutor.py**

`src/hey_jude/services/substitutor.py`:
```python
import json
import re

import httpx

from hey_jude.config import Settings
from hey_jude.models import ChatMessage, DetectedEntity, SubstitutionResult


def _build_deterministic_replacements(
    entities: list[DetectedEntity],
    strategies: dict[str, str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    counter = 0
    for entity in entities:
        if strategies.get(entity.entity_type) != "deterministic":
            continue
        if entity.text in mapping:
            continue
        counter += 1
        if entity.entity_type == "EMAIL_ADDRESS":
            mapping[entity.text] = f"user_{counter}@example.com"
        elif entity.entity_type == "PHONE_NUMBER":
            mapping[entity.text] = f"555-{100 + counter:04d}"
        else:
            mapping[entity.text] = f"REDACTED_{entity.entity_type}_{counter}"
    return mapping


def _build_prompt(entities: list[DetectedEntity], query: str) -> str:
    entity_list = json.dumps(
        [{"text": e.text, "type": e.entity_type} for e in entities],
        indent=2,
    )
    return f"""<task>
Analyze this legal professional's query for sensitive entity handling.
</task>

<entities>
{entity_list}
</entities>

<query>
{query}
</query>

<instructions>
1. Classify sensitivity: "low" if entity replacement alone prevents identification, "high" if structural patterns or relationships could de-anonymize.
2. For each entity, provide a brief context descriptor (what kind of entity it is, without identifying it).
3. Generate fictional replacement names that preserve the entity's role and domain. Always use fictional names, never descriptive phrases.
4. If high sensitivity: also rephrase the query to obscure identifying structural patterns while preserving the legal question's intent.
5. If you are unsure about the appropriate anonymization strategy, set needs_clarification to true and provide a clarification question.
6. Return ONLY a JSON object with these keys: sensitivity, reasoning, mapping, context_descriptors, sanitized_text, needs_clarification, clarification_question
</instructions>"""


def _parse_llm_response(raw: str) -> dict:
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse local LLM response as JSON: {e}") from e


async def _call_local_llm(prompt: str, settings: Settings) -> str:
    url = f"{settings.local_llm_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.local_llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _apply_mapping_to_text(text: str, mapping: dict[str, str]) -> str:
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    for original in sorted_keys:
        replacement = mapping[original]
        text = text.replace(original, replacement)
    return text


async def substitute_entities(
    messages: list[ChatMessage],
    entities: list[DetectedEntity],
    settings: Settings,
) -> SubstitutionResult:
    deterministic_mapping = _build_deterministic_replacements(
        entities, settings.entity_strategies
    )

    llm_entities = [
        e for e in entities
        if settings.entity_strategies.get(e.entity_type) == "llm"
    ]

    llm_mapping: dict[str, str] = {}
    context_descriptors: dict[str, str] = {}
    sensitivity = "low"
    needs_clarification = False
    clarification_question = None

    if llm_entities:
        user_text = " ".join(
            m.content for m in messages if m.role in ("user", "assistant")
        )
        if settings.max_context_window and len(user_text) > settings.max_context_window:
            user_text = user_text[:settings.max_context_window]

        prompt = _build_prompt(llm_entities, user_text)
        raw_response = await _call_local_llm(prompt, settings)

        try:
            parsed = _parse_llm_response(raw_response)
        except ValueError:
            retry_prompt = prompt + "\n\nIMPORTANT: Return ONLY valid JSON. No markdown, no explanation."
            raw_response = await _call_local_llm(retry_prompt, settings)
            parsed = _parse_llm_response(raw_response)

        llm_mapping = parsed.get("mapping", {})
        context_descriptors = parsed.get("context_descriptors", {})
        sensitivity = parsed.get("sensitivity", "low")
        needs_clarification = parsed.get("needs_clarification", False)
        clarification_question = parsed.get("clarification_question")

        if settings.always_full_anonymization:
            sensitivity = "high"

    full_mapping = {**deterministic_mapping, **llm_mapping}
    reverse_mapping = {v: k for k, v in full_mapping.items()}

    sanitized_messages = []
    for msg in messages:
        if msg.role == "system" and settings.passthrough_system_messages:
            sanitized_messages.append(msg)
        else:
            sanitized_content = _apply_mapping_to_text(msg.content, full_mapping)
            sanitized_messages.append(
                ChatMessage(role=msg.role, content=sanitized_content)
            )

    return SubstitutionResult(
        mapping=full_mapping,
        reverse_mapping=reverse_mapping,
        context_descriptors=context_descriptors,
        sanitized_messages=sanitized_messages,
        sensitivity=sensitivity,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_substitutor.py -v`
Expected: All 6 tests PASS (only unit-level functions tested, no local LLM needed).

- [ ] **Step 5: Commit**

```bash
git add src/hey_jude/services/substitutor.py tests/test_substitutor.py
git commit -m "feat: three-phase substitutor with local LLM and deterministic fallbacks"
```

---

## Task 7: LiteLLM Router

**Files:**
- Create: `src/hey_jude/services/router.py`
- Create: `tests/test_router.py`

- [ ] **Step 1: Write failing tests**

`tests/test_router.py`:
```python
import pytest
from hey_jude.models import ChatMessage
from hey_jude.services.router import _build_messages_with_preamble


def test_preamble_injected_as_system_message():
    messages = [
        ChatMessage(role="user", content="Tell me about Pinnacle Systems"),
    ]
    context_descriptors = {
        "Pinnacle Systems": "major technology corporation",
        "Vertex Holdings": "social media conglomerate",
    }
    result = _build_messages_with_preamble(messages, context_descriptors)
    assert result[0]["role"] == "system"
    assert "Pinnacle Systems: major technology corporation" in result[0]["content"]
    assert "Vertex Holdings: social media conglomerate" in result[0]["content"]
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "Tell me about Pinnacle Systems"


def test_preamble_prepended_before_existing_system():
    messages = [
        ChatMessage(role="system", content="You are a legal assistant."),
        ChatMessage(role="user", content="Tell me about Pinnacle Systems"),
    ]
    context_descriptors = {"Pinnacle Systems": "major technology corporation"}
    result = _build_messages_with_preamble(messages, context_descriptors)
    assert result[0]["role"] == "system"
    assert "Pinnacle Systems" in result[0]["content"]
    assert result[1]["role"] == "system"
    assert result[1]["content"] == "You are a legal assistant."


def test_empty_descriptors_no_preamble():
    messages = [
        ChatMessage(role="user", content="Hello"),
    ]
    result = _build_messages_with_preamble(messages, {})
    assert len(result) == 1
    assert result[0]["role"] == "user"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_router.py -v`
Expected: ImportError — `hey_jude.services.router` does not exist.

- [ ] **Step 3: Implement router.py**

`src/hey_jude/services/router.py`:
```python
import litellm

from hey_jude.models import ChatMessage


def _build_messages_with_preamble(
    messages: list[ChatMessage],
    context_descriptors: dict[str, str],
) -> list[dict]:
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    if not context_descriptors:
        return msg_dicts
    lines = [f"- {name}: {desc}" for name, desc in context_descriptors.items()]
    preamble = "[Context: For this query, the following entities are referenced:\n" + "\n".join(lines) + "]"
    preamble_msg = {"role": "system", "content": preamble}
    return [preamble_msg] + msg_dicts


async def route_completion(
    messages: list[ChatMessage],
    context_descriptors: dict[str, str],
    model: str,
    **kwargs,
) -> dict:
    final_messages = _build_messages_with_preamble(messages, context_descriptors)
    response = await litellm.acompletion(
        model=model,
        messages=final_messages,
        **kwargs,
    )
    return response.model_dump()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hey_jude/services/router.py tests/test_router.py
git commit -m "feat: LiteLLM router with context preamble injection"
```

---

## Task 8: FastAPI App, Routes, and Auth

**Files:**
- Create: `src/hey_jude/main.py`
- Create: `src/hey_jude/routes.py`
- Create: `tests/test_routes.py`

- [ ] **Step 1: Write failing tests**

`tests/test_routes.py`:
```python
import pytest
from httpx import ASGITransport, AsyncClient

from hey_jude.main import create_app
from hey_jude.config import Settings


@pytest.fixture
def test_settings():
    return Settings(
        redis_url="redis://localhost:6379/1",
        redis_ttl_seconds=60,
        api_key="sk-test-key",
    )


@pytest.fixture
async def app(test_settings):
    application = create_app(test_settings)
    yield application


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "redis" in data


async def test_auth_missing_key(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 401


async def test_auth_wrong_key(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


async def test_auth_valid_key_bearer(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "The weather is nice."}],
        },
        headers={"Authorization": "Bearer sk-test-key"},
    )
    # Should not be 401 — may be 503 if Redis/LLM not available, but auth passed
    assert resp.status_code != 401


async def test_malformed_request(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={"not": "valid"},
        headers={"X-API-Key": "sk-test-key"},
    )
    assert resp.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_routes.py -v`
Expected: ImportError — `hey_jude.main` does not exist.

- [ ] **Step 3: Implement main.py**

`src/hey_jude/main.py`:
```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from hey_jude.config import Settings, settings as default_settings
from hey_jude.redis_client import RedisClient
from hey_jude.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis_client = RedisClient(app.state.settings.redis_url)
    await app.state.redis_client.connect()
    yield
    await app.state.redis_client.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="Hey Jude",
        description="Context-preserving pseudonymization gateway",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings or default_settings
    app.include_router(router)
    return app


app = create_app()
```

- [ ] **Step 4: Implement routes.py**

`src/hey_jude/routes.py`:
```python
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from httpx import ConnectError as HttpxConnectError

from hey_jude.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    HeyJudeMetadata,
)
from hey_jude.services.detector import detect_entities
from hey_jude.services.mapper import reverse_map_text
from hey_jude.services.router import route_completion
from hey_jude.services.substitutor import substitute_entities

router = APIRouter()


def _extract_api_key(request: Request) -> str | None:
    key = request.headers.get("X-API-Key")
    if key:
        return key
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


@router.get("/health")
async def health(request: Request):
    redis_ok = False
    try:
        redis_ok = await request.app.state.redis_client.health_check()
    except Exception:
        pass
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": redis_ok,
    }


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    settings = request.app.state.settings
    redis_client = request.app.state.redis_client

    api_key = _extract_api_key(request)
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    request_id = str(uuid.uuid4())

    all_entities = []
    for msg in body.messages:
        if msg.role == "system" and settings.passthrough_system_messages:
            continue
        entities = await detect_entities(msg.content, settings)
        all_entities.extend(entities)

    if not all_entities:
        completion_kwargs = {}
        if body.temperature is not None:
            completion_kwargs["temperature"] = body.temperature
        if body.max_tokens is not None:
            completion_kwargs["max_tokens"] = body.max_tokens
        if body.top_p is not None:
            completion_kwargs["top_p"] = body.top_p

        result = await route_completion(
            messages=body.messages,
            context_descriptors={},
            model=settings.external_llm_model,
            **completion_kwargs,
        )
        result["heyjude_metadata"] = {
            "request_id": request_id,
            "entities_detected": 0,
            "sensitivity": "none",
            "status": "completed",
        }
        return result

    try:
        redis_ok = await redis_client.health_check()
        if not redis_ok:
            raise HTTPException(status_code=503, detail="Redis unavailable")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    try:
        sub_result = await substitute_entities(body.messages, all_entities, settings)
    except HttpxConnectError:
        raise HTTPException(status_code=503, detail="Local LLM unavailable")
    except ValueError:
        raise HTTPException(status_code=502, detail="Local LLM returned unparseable response")

    if sub_result.needs_clarification and settings.allow_clarification_requests:
        await redis_client.store_request(
            request_id,
            body.model_dump(),
            settings.redis_ttl_seconds,
        )
        return ChatCompletionResponse(
            id=f"heyjude-clarify-{request_id}",
            created=int(time.time()),
            model=body.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=sub_result.clarification_question or "Could you clarify your request?",
                    ),
                    finish_reason="stop",
                )
            ],
            heyjude_metadata=HeyJudeMetadata(
                request_id=request_id,
                entities_detected=len(all_entities),
                sensitivity=sub_result.sensitivity,
                status="clarification_needed",
            ),
        )

    await redis_client.store_mapping(
        request_id,
        sub_result.mapping,
        settings.redis_ttl_seconds,
    )

    completion_kwargs = {}
    if body.temperature is not None:
        completion_kwargs["temperature"] = body.temperature
    if body.max_tokens is not None:
        completion_kwargs["max_tokens"] = body.max_tokens
    if body.top_p is not None:
        completion_kwargs["top_p"] = body.top_p

    external_response = await route_completion(
        messages=sub_result.sanitized_messages,
        context_descriptors=sub_result.context_descriptors,
        model=settings.external_llm_model,
        **completion_kwargs,
    )

    for choice in external_response.get("choices", []):
        content = choice.get("message", {}).get("content", "")
        choice["message"]["content"] = reverse_map_text(content, sub_result.reverse_mapping)

    external_response["heyjude_metadata"] = {
        "request_id": request_id,
        "entities_detected": len(all_entities),
        "sensitivity": sub_result.sensitivity,
        "status": "completed",
    }

    return external_response
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_routes.py -v`
Expected: All 5 tests PASS (health, auth missing, auth wrong, auth valid bearer, malformed request).

- [ ] **Step 6: Commit**

```bash
git add src/hey_jude/main.py src/hey_jude/routes.py tests/test_routes.py
git commit -m "feat: FastAPI app with /v1/chat/completions endpoint and API key auth"
```

---

## Task 9: Docker and Deployment Files

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create Dockerfile**

`Dockerfile`:
```dockerfile
FROM python:3.11-slim AS base

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

# Presidio requires downloading an spaCy model
RUN python -m spacy download en_core_web_lg

EXPOSE 8000

CMD ["uvicorn", "hey_jude.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create docker-compose.yml**

`docker-compose.yml`:
```yaml
services:
  gateway:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      redis:
        condition: service_healthy
    environment:
      - REDIS_URL=redis://redis:6379/0

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

- [ ] **Step 3: Verify docker-compose config is valid**

Run: `docker compose config`
Expected: Prints resolved config without errors.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: Dockerfile and docker-compose for gateway + Redis"
```

---

## Task 10: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

`README.md`:
```markdown
# Hey Jude

Context-preserving pseudonymization gateway for law firms. Intercepts LLM prompts, replaces sensitive entities with contextually accurate fictional equivalents using a local LLM, routes sanitized prompts to an external enterprise LLM, and reverses substitutions in the response.

## How It Works

1. **Detect** - Presidio scans prompts for PII entities (names, organizations, emails, phone numbers)
2. **Analyze** - Local LLM classifies query sensitivity and generates context descriptors
3. **Substitute** - Entities replaced with fictional names; high-sensitivity queries get structural abstraction
4. **Route** - Sanitized prompt sent to external LLM (Azure, OpenAI, etc.) via LiteLLM
5. **Reverse** - Synthetic entities in the response replaced back with originals
6. **Return** - Clean, de-anonymized response returned to the user

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A local LLM running an OpenAI-compatible API (e.g., [Ollama](https://ollama.ai), [LM Studio](https://lmstudio.ai))
- Access to an external LLM (Azure OpenAI, OpenAI, etc.)

### Setup

```bash
cp .env.example .env
# Edit .env with your configuration
docker compose up
```

The gateway is available at `http://localhost:8000`.

### Usage

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "What IP cases involve Microsoft and Apple?"}
    ]
  }'
```

## Configuration

All configuration via environment variables. See `.env.example` for the full list.

### Core Settings

| Variable | Description | Default |
|----------|------------|---------|
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `REDIS_TTL_SECONDS` | Mapping expiry (seconds) | `3600` |
| `LOCAL_LLM_URL` | Local LLM endpoint | `http://localhost:11434/v1` |
| `LOCAL_LLM_MODEL` | Local model name | `llama3.2` |
| `EXTERNAL_LLM_MODEL` | LiteLLM model ID | `azure/gpt-4o` |
| `API_KEY` | Gateway API key | - |

### Entity Strategies

Configure per-entity-type replacement strategy:

| Strategy | Behavior |
|----------|----------|
| `llm` | Local LLM generates context-aware fictional replacement |
| `deterministic` | Pattern-based replacement (emails, phone numbers) |

### Guardrails

| Toggle | Default | Description |
|--------|---------|------------|
| `ALWAYS_FULL_ANONYMIZATION` | `false` | Force high-sensitivity mode for all queries |
| `ANONYMIZE_PRODUCT_NAMES` | `true` | Replace product names alongside org names |
| `ABSTRACT_RELATIONSHIPS` | `true` | Rephrase structural patterns in high-sensitivity queries |
| `ALLOW_CLARIFICATION_REQUESTS` | `true` | Allow gateway to ask for clarification |

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests that interact with Redis require a running Redis instance on `localhost:6379`.

## API

### POST /v1/chat/completions

OpenAI-compatible chat completion endpoint. Responses include a `heyjude_metadata` field with anonymization details.

### GET /health

Returns gateway health status including Redis connectivity.

## License

MIT
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with quickstart, configuration, and API reference"
```

---

## Task 11: Run Full Test Suite and Verify

- [ ] **Step 1: Ensure Redis is running**

Run: `docker run -d --name heyjude-test-redis -p 6379:6379 redis:7-alpine`
(Skip if Redis already running.)

- [ ] **Step 2: Install project**

Run: `pip install -e ".[dev]"`

- [ ] **Step 3: Run all tests**

Run: `pytest -v`
Expected: All tests pass across test_models, test_redis_client, test_detector, test_substitutor, test_mapper, test_router, test_routes.

- [ ] **Step 4: Fix any failures**

Address test failures. Common issues:
- Presidio may need `python -m spacy download en_core_web_lg` for the NER model
- Redis tests need a running Redis on port 6379

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address test suite issues from integration run"
```

---

## Verification Checklist

After all tasks complete:

1. `docker compose up` starts gateway + Redis without errors
2. `GET /health` returns `{"status": "ok", "redis": true}`
3. `POST /v1/chat/completions` without API key returns 401
4. `POST /v1/chat/completions` with valid key and entity-containing text:
   - Detects entities
   - Calls local LLM for substitution
   - Routes sanitized text to external LLM
   - Reverses entities in response
   - Returns `heyjude_metadata` with request_id, entity count, sensitivity
5. `POST /v1/chat/completions` with no entities bypasses anonymization
6. `pytest -v` all green
