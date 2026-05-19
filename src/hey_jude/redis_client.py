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
