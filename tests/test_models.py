from hey_jude.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DetectedEntity,
    HeyJudeMetadata,
    SubstitutionResult,
    Usage,
)


def test_chat_message_valid():
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


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
        clarification_question=None,
    )
    assert result.mapping["Microsoft"] == "Pinnacle Systems"
    assert result.reverse_mapping["Pinnacle Systems"] == "Microsoft"
