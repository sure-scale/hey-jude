from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    request_id: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class ChoiceMessage(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class HeyJudeMetadata(BaseModel):
    request_id: str
    entities_detected: int
    sensitivity: str
    status: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    heyjude_metadata: HeyJudeMetadata | None = None


@dataclass
class DetectedEntity:
    text: str
    entity_type: str
    start: int
    end: int
    score: float


@dataclass
class SubstitutionResult:
    mapping: dict[str, str]
    reverse_mapping: dict[str, str]
    context_descriptors: dict[str, str]
    sanitized_messages: list[ChatMessage]
    sensitivity: str
    needs_clarification: bool
    clarification_question: str | None = None


@dataclass
class FoundEntity:
    text: str
    entity_type: str
    action: str
    replacement: str | None
    reason: str | None


@dataclass
class AnonymizationResult:
    mapping: dict[str, str]
    reverse_mapping: dict[str, str]
    context_descriptors: dict[str, str]
    sanitized_messages: list[ChatMessage]
    sensitivity: str
    entities_found: list[FoundEntity]


@dataclass
class SafetyNetResult:
    passed: bool
    leaked_entities: list[DetectedEntity]
    auto_replaced: int
