import re

import litellm

from hey_jude.models import ChatMessage

_DESCRIPTOR_STOPWORDS = {"a", "an", "the", "of", "for", "this", "that", "some"}


def _placeholder_words(placeholder: str) -> set[str]:
    """Tokenize a placeholder (COMPANY_01) into its meaningful words."""
    return {
        part
        for part in re.split(r"[_\s]+", placeholder.casefold())
        if part and not part.isdigit()
    }


def _descriptor_is_redundant(placeholder: str, descriptor: str) -> bool:
    """True if the descriptor adds nothing the placeholder doesn't already say.

    Prevents the category leaking twice — e.g. placeholder INVESTMENT_BANK_01
    with descriptor "investment bank". If every meaningful descriptor word is
    already carried by the placeholder, the descriptor is pure redundant signal.
    """
    desc_words = {
        word
        for word in re.split(r"[^a-z0-9]+", descriptor.casefold())
        if word and word not in _DESCRIPTOR_STOPWORDS
    }
    if not desc_words:
        return True
    return desc_words <= _placeholder_words(placeholder)


def _build_messages_with_preamble(
    messages: list[ChatMessage],
    context_descriptors: dict[str, str],
    sensitivity: str = "low",
) -> list[dict]:
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    # On high sensitivity the descriptors are a net inference risk: any detail
    # that aids the downstream model also aids a re-identifier. Drop them.
    if sensitivity == "high" or not context_descriptors:
        return msg_dicts
    lines = [
        f"- {name}: {desc}"
        for name, desc in context_descriptors.items()
        if desc and not _descriptor_is_redundant(name, desc)
    ]
    if not lines:
        return msg_dicts
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
    sensitivity: str = "low",
    **kwargs,
) -> dict:
    final_messages = _build_messages_with_preamble(
        messages, context_descriptors, sensitivity
    )
    response = await litellm.acompletion(
        model=model,
        messages=final_messages,
        **kwargs,
    )
    return response.model_dump()
