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
