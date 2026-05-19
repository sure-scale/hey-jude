import dotenv
dotenv.load_dotenv()

from pydantic import Field
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

    external_llm_model: str = "ollama_chat/qwen3.5:4b"
    external_llm_api_base: str | None = "http://localhost:11434"

    api_key: str = "sk-heyjude-dev"

    presidio_entities: list[str] = Field(
        default=["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER"]
    )
    entity_strategies: dict[str, str] = Field(
        default={
            "PERSON": "llm",
            "ORGANIZATION": "llm",
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


settings = Settings()
