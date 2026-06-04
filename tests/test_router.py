import pytest
from hey_jude.config import Settings
from hey_jude.models import ChatMessage, IrreducibilityAssessment
from hey_jude.services.router import (
    IN_JURISDICTION_SENSITIVE,
    SOVEREIGN_TIERS,
    US_PSEUDONYMIZED,
    _build_messages_with_preamble,
    _descriptor_is_redundant,
    select_route,
)


def _settings(**overrides):
    return Settings(
        external_llm_model="us-frontier",
        external_llm_api_base="https://us.example",
        external_llm_model_sensitive="sovereign-oss",
        external_llm_model_sensitive_api_base="http://localhost:11434",
        **overrides,
    )


_REDUCIBLE = IrreducibilityAssessment(irreducible=False, reason=None, risk=0.1)
_IRREDUCIBLE = IrreducibilityAssessment(
    irreducible=True, reason="NAMED_ENTITY_ESSENTIAL", risk=0.9
)


def test_sovereign_tiers_definition():
    assert IN_JURISDICTION_SENSITIVE in SOVEREIGN_TIERS
    assert US_PSEUDONYMIZED not in SOVEREIGN_TIERS


def test_reducible_routes_to_us_frontier():
    decision = select_route(_REDUCIBLE, _settings(irreducible_policy="BLOCK"))
    assert decision.model == "us-frontier"
    assert decision.api_base == "https://us.example"
    assert decision.tier == US_PSEUDONYMIZED
    assert decision.tier not in SOVEREIGN_TIERS
    assert decision.action == "allow"


def test_irreducible_block_refuses_egress():
    decision = select_route(_IRREDUCIBLE, _settings(irreducible_policy="BLOCK"))
    assert decision.model is None
    assert decision.tier == IN_JURISDICTION_SENSITIVE
    assert decision.tier in SOVEREIGN_TIERS
    assert decision.action == "block"


def test_irreducible_allow_opts_into_us():
    decision = select_route(_IRREDUCIBLE, _settings(irreducible_policy="ALLOW"))
    assert decision.model == "us-frontier"
    assert decision.tier == US_PSEUDONYMIZED
    assert decision.tier not in SOVEREIGN_TIERS
    assert decision.action == "allow"


@pytest.mark.parametrize(
    "policy,action", [("WARN", "warn"), ("ASK", "ask")]
)
def test_irreducible_warn_ask_divert_to_sovereign(policy, action):
    decision = select_route(_IRREDUCIBLE, _settings(irreducible_policy=policy))
    assert decision.model == "sovereign-oss"
    assert decision.api_base == "http://localhost:11434"
    assert decision.tier == IN_JURISDICTION_SENSITIVE
    assert decision.tier in SOVEREIGN_TIERS
    assert decision.action == action


def test_override_supersedes_settings_policy():
    # Settings say ALLOW (would egress to US), header override forces BLOCK.
    decision = select_route(
        _IRREDUCIBLE, _settings(irreducible_policy="ALLOW"), override="BLOCK"
    )
    assert decision.model is None
    assert decision.action == "block"


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


def test_descriptor_redundant_when_words_carried_by_placeholder():
    assert _descriptor_is_redundant("INVESTMENT_BANK_01", "investment bank")
    assert _descriptor_is_redundant("COMPANY_01", "a company")


def test_descriptor_not_redundant_when_it_adds_words():
    assert not _descriptor_is_redundant(
        "COMPANY_01", "major technology corporation"
    )


def test_redundant_descriptor_filtered_from_preamble():
    messages = [ChatMessage(role="user", content="Tell me about INVESTMENT_BANK_01")]
    result = _build_messages_with_preamble(
        messages, {"INVESTMENT_BANK_01": "investment bank"}
    )
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_high_sensitivity_drops_all_descriptors():
    messages = [ChatMessage(role="user", content="Tell me about COMPANY_01")]
    context_descriptors = {"COMPANY_01": "major technology corporation"}
    result = _build_messages_with_preamble(
        messages, context_descriptors, sensitivity="high"
    )
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Tell me about COMPANY_01"
