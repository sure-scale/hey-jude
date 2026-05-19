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
    preamble = (
        "[Context: For this query, the following entities are referenced:\n"
        + "\n".join(lines)
        + "]"
    )
    preamble_msg = {"role": "system", "content": preamble}
    return [preamble_msg] + msg_dicts


async def route_completion(
    messages: list[ChatMessage],
    context_descriptors: dict[str, str],
    model: str,
    **kwargs,
) -> dict:
    final_messages = _build_messages_with_preamble(
        messages, context_descriptors
    )
    response = await litellm.acompletion(
        model=model,
        messages=final_messages,
        **kwargs,
    )
    return response.model_dump()
