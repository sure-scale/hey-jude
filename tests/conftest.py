import os
from pathlib import Path

import pytest


os.environ.setdefault("TLDEXTRACT_CACHE", "/private/tmp/codex-cache/tldextract")
os.environ.setdefault("LOCAL_LLM_MODEL", "qwen3.5:4b")
os.environ.setdefault("EXTERNAL_LLM_MODEL", "ollama_chat/qwen3.5:4b")
os.environ.setdefault("EXTERNAL_LLM_API_BASE", "http://localhost:11434")
os.environ.setdefault(
    "ENTITY_STRATEGIES",
    '{"PERSON":"llm","ORGANIZATION":"llm","EMAIL_ADDRESS":"deterministic","PHONE_NUMBER":"deterministic"}',
)

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
    # Try flushing DB, but if Redis is down, we handle or fail test gracefully
    try:
        await client._redis.flushdb()
    except Exception:
        pass
    yield client
    try:
        await client._redis.flushdb()
    except Exception:
        pass
    await client.close()


@pytest.fixture(autouse=True)
def tldextract_cache_dir(monkeypatch):
    cache_dir = Path("/private/tmp/codex-cache/tldextract")
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TLDEXTRACT_CACHE", str(cache_dir))
