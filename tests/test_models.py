from hey_jude.models import (
    AnonymizationResult,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DetectedEntity,
    FoundEntity,
    HeyJudeMetadata,
    IrreducibilityAssessment,
    SafetyNetResult,
    SubstitutionResult,
    Usage,
)


def test_chat_message_valid():
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_chat_message_accepts_openai_content_parts():
    msg = ChatMessage(
        role="user",
        content=[
            {"type": "text", "text": "Review this."},
            {"type": "input_file", "filename": "memo.pdf", "file_data": "abc"},
        ],
    )
    assert msg.content[0]["text"] == "Review this."


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
        irreducibility=IrreducibilityAssessment(
            irreducible=False, reason=None, risk=0.0
        ),
        clarification_question=None,
    )
    assert result.mapping["Microsoft"] == "Pinnacle Systems"
    assert result.reverse_mapping["Pinnacle Systems"] == "Microsoft"
    assert result.irreducibility.irreducible is False


def test_found_entity_replace():
    e = FoundEntity(
        text="Microsoft",
        entity_type="ORGANIZATION",
        action="replace",
        replacement="SOFTWARE_COMPANY_01",
        reason="real company",
    )
    assert e.action == "replace"
    assert e.replacement == "SOFTWARE_COMPANY_01"


def test_found_entity_keep():
    e = FoundEntity(
        text="Purchaser",
        entity_type="DEFINED_TERM",
        action="keep",
        replacement=None,
        reason="legal defined term",
    )
    assert e.action == "keep"
    assert e.replacement is None


def test_anonymization_result():
    result = AnonymizationResult(
        mapping={"Microsoft": "SOFTWARE_COMPANY_01"},
        reverse_mapping={"SOFTWARE_COMPANY_01": "Microsoft"},
        context_descriptors={"SOFTWARE_COMPANY_01": "tech company"},
        sanitized_messages=[ChatMessage(role="user", content="SOFTWARE_COMPANY_01")],
        sensitivity="low",
        entities_found=[
            FoundEntity(
                text="Microsoft",
                entity_type="ORGANIZATION",
                action="replace",
                replacement="SOFTWARE_COMPANY_01",
                reason="real company",
            )
        ],
        irreducibility=IrreducibilityAssessment(
            irreducible=False, reason=None, risk=0.0
        ),
    )
    assert result.mapping["Microsoft"] == "SOFTWARE_COMPANY_01"
    assert len(result.entities_found) == 1


def test_safety_net_result_passed():
    result = SafetyNetResult(passed=True, leaked_entities=[], auto_replaced=0)
    assert result.passed is True


def test_safety_net_result_failed():
    result = SafetyNetResult(
        passed=False,
        leaked_entities=[
            DetectedEntity(text="John", entity_type="PERSON", start=0, end=4, score=0.9)
        ],
        auto_replaced=0,
    )
    assert result.passed is False
    assert len(result.leaked_entities) == 1
