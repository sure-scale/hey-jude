import dotenv
dotenv.load_dotenv()

from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 3600

    local_llm_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "qwen3.5:4b"
    local_llm_api_key: str = ""

    external_llm_model: str = "ollama_chat/qwen3.5:4b"
    external_llm_api_base: str | None = "http://localhost:11434"

    api_key: str = "sk-heyjude-dev"

    presidio_entities: list[str] = Field(
        default=["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER"]
    )
    entity_strategies: dict[str, str] = Field(
        default={
            "PERSON": "placeholder",
            "ORGANIZATION": "placeholder",
            "EMAIL_ADDRESS": "deterministic",
            "PHONE_NUMBER": "deterministic",
        }
    )

    always_full_anonymization: bool = False
    anonymize_product_names: bool = True
    abstract_relationships: bool = True
    passthrough_system_messages: bool = False
    max_context_window: int = 500
    allow_clarification_requests: bool = True

    anonymization_mode: Literal["llm", "mechanical"] = "llm"
    safety_net_strictness: Literal["off", "warn", "strict"] = "warn"
    document_unreadable_action: Literal["reject", "warn", "skip"] = "reject"
    anonymization_prompt_path: str = "prompts/anonymize.txt"

    # Runtime re-identification critic. After anonymizing a "high" sensitivity
    # request, run one blind re-identification pass over the sanitized output and
    # broaden the descriptors of any entity the critic pins down. Bounds cost to
    # a single extra local-LLM call, and only on high-sensitivity requests.
    reid_critic_enabled: bool = True
    reid_critic_prompt_path: str = "prompts/reid_critic.txt"
    reid_critic_threshold: float = 0.6

    custom_recognizers_path: str | None = None
    known_entities_path: str | None = None

    # Request-level audit logging. Off by default. The log is tamper-evident
    # (hash-chained JSONL) and meant to be kept internal. Default content level
    # stores no raw PII — only metadata plus content digests.
    audit_enabled: bool = False
    audit_destination: str = "stdout"  # "stdout" or a file path
    audit_content_level: Literal["metadata", "anonymized", "full"] = "metadata"
    audit_rotation: Literal["none", "daily", "monthly"] = "monthly"
    audit_hmac_key: str | None = None
    audit_failure_mode: Literal["ignore", "fail"] = "ignore"
    audit_log_ip: bool = True
    audit_trust_forwarded_for: bool = False
    audit_matter_header: str = "X-Heyjude-Matter-Id"
    audit_actor_header: str | None = None  # opt-in; e.g. "X-Heyjude-Actor"

    # Populated from custom_recognizers_path; not read from the environment.
    custom_recognizer_specs: list[Any] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def _load_custom_recognizers(self) -> "Settings":
        if not self.custom_recognizers_path:
            return self

        from hey_jude.services.recognizers import load_recognizer_specs

        specs = load_recognizer_specs(self.custom_recognizers_path)
        self.custom_recognizer_specs = specs

        # Custom entity types must be allow-listed for the analyzer to surface
        # them, and need a substitution strategy for mechanical mode.
        entities = list(self.presidio_entities)
        strategies = dict(self.entity_strategies)
        for spec in specs:
            if spec.entity_type not in entities:
                entities.append(spec.entity_type)
            strategies.setdefault(spec.entity_type, spec.strategy)
        self.presidio_entities = entities
        self.entity_strategies = strategies
        return self


settings = Settings()
