import json
import asyncio
import pytest


async def test_store_and_get_mapping(redis_client):
    # Skip if Redis is not running
    if not await redis_client.health_check():
        pytest.skip("Redis is not running")
        
    mapping = {"Microsoft": "Pinnacle Systems", "John": "James"}
    await redis_client.store_mapping("req-1", mapping, ttl=60)
    result = await redis_client.get_mapping("req-1")
    assert result == mapping


async def test_get_mapping_missing_key(redis_client):
    if not await redis_client.health_check():
        pytest.skip("Redis is not running")
        
    result = await redis_client.get_mapping("nonexistent")
    assert result is None


async def test_mapping_ttl_enforced(redis_client):
    if not await redis_client.health_check():
        pytest.skip("Redis is not running")
        
    await redis_client.store_mapping("req-2", {"a": "b"}, ttl=1)
    await asyncio.sleep(1.1)
    result = await redis_client.get_mapping("req-2")
    assert result is None


async def test_store_and_get_request(redis_client):
    if not await redis_client.health_check():
        pytest.skip("Redis is not running")
        
    request_data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    await redis_client.store_request("req-3", request_data, ttl=60)
    result = await redis_client.get_request("req-3")
    assert result == request_data


async def test_get_request_missing_key(redis_client):
    if not await redis_client.health_check():
        pytest.skip("Redis is not running")
        
    result = await redis_client.get_request("nonexistent")
    assert result is None


async def test_health_check(redis_client):
    # Verify health check is a boolean reflecting if Redis is running
    is_up = await redis_client.health_check()
    assert isinstance(is_up, bool)
