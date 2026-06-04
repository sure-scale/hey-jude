import re
from dataclasses import dataclass

import litellm

from hey_jude.models import ChatMessage, IrreducibilityAssessment

_DESCRIPTOR_STOPWORDS = {"a", "an", "the", "of", "for", "this", "that", "some"}

# Route destinations tiered by ACTUAL jurisdiction, not by "managed = safe".
# US_PSEUDONYMIZED is a US-jurisdiction frontier model that only ever sees
# pseudonyms. IN_JURISDICTION_SENSITIVE is self-hosted OSS on compute not subject
# to US compulsion — the only sovereign tier here. Azure OpenAI / AWS Bedrock are
# US jurisdiction (CLOUD Act reaches the US parent regardless of data residency)
# and would NOT belong in SOVEREIGN_TIERS if added later.
US_PSEUDONYMIZED = "us-ok-pseudonymized"
IN_JURISDICTION_SENSITIVE = "in-jurisdiction-sensitive"
SOVEREIGN_TIERS = {IN_JURISDICTION_SENSITIVE}


@dataclass
class RouteDecision:
    """The single egress decision: where a request goes and what to tell the user.

    model is None only when action is "block" (no egress at all).
    """

    model: str | None
    api_base: str | None
    tier: str
    action: str  # allow | warn | ask | block


def select_route(
    assessment: IrreducibilityAssessment,
    settings,
    override: str | None = None,
) -> RouteDecision:
    """Decide egress from residual risk and policy — the one gate for the system.

    Reducible requests go to the US frontier model with only pseudonyms exposed,
    exactly as before; sensitivity alone never diverts. Only an *irreducible*
    request — one that leaks identity regardless of masking — is governed by the
    policy: BLOCK refuses, ASK/WARN divert to the sovereign in-jurisdiction
    model, ALLOW sends to the US model anyway as an explicit opt-in.
    """
    if not assessment.irreducible:
        return RouteDecision(
            model=settings.external_llm_model,
            api_base=settings.external_llm_api_base,
            tier=US_PSEUDONYMIZED,
            action="allow",
        )

    policy = override or settings.irreducible_policy
    if policy == "ALLOW":
        return RouteDecision(
            model=settings.external_llm_model,
            api_base=settings.external_llm_api_base,
            tier=US_PSEUDONYMIZED,
            action="allow",
        )
    if policy == "BLOCK":
        return RouteDecision(
            model=None,
            api_base=None,
            tier=IN_JURISDICTION_SENSITIVE,
            action="block",
        )
    return RouteDecision(
        model=settings.external_llm_model_sensitive,
        api_base=settings.external_llm_model_sensitive_api_base,
        tier=IN_JURISDICTION_SENSITIVE,
        action="ask" if policy == "ASK" else "warn",
    )


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
