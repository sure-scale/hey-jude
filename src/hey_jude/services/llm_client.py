import json
import re
from urllib.parse import urlparse

import httpx

from hey_jude.config import Settings


def ollama_native_chat_url(local_llm_url: str) -> str | None:
    parsed = urlparse(local_llm_url)
    if parsed.port != 11434:
        return None

    path = parsed.path.rstrip("/")
    if path not in {"", "/v1"}:
        return None

    return f"{parsed.scheme}://{parsed.netloc}/api/chat"


def parse_llm_response(raw: str) -> dict:
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse local LLM response as JSON: {e}") from e


async def _call_ollama_native_llm(prompt: str, settings: Settings, url: str) -> str:
    payload = {
        "model": settings.local_llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 4096,
        },
    }
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


async def call_local_llm(prompt: str, settings: Settings) -> str:
    ollama_url = ollama_native_chat_url(settings.local_llm_url)
    if ollama_url:
        return await _call_ollama_native_llm(prompt, settings, ollama_url)

    url = f"{settings.local_llm_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.local_llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 32768,
    }
    headers = {}
    if settings.local_llm_api_key:
        headers["api-key"] = settings.local_llm_api_key
        headers["Authorization"] = f"Bearer {settings.local_llm_api_key}"
    async with httpx.AsyncClient(timeout=300.0, trust_env=False) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning_content")
        if not content:
            raise ValueError(
                f"Local LLM returned an empty completion: {data['choices'][0]!r}"
            )
        return content
