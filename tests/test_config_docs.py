from pathlib import Path

from hey_jude.config import Settings


ROOT = Path(__file__).resolve().parents[1]


def test_anonymization_mode_defaults_to_llm():
    s = Settings()
    assert s.anonymization_mode == "llm"


def test_safety_net_strictness_defaults_to_warn():
    s = Settings()
    assert s.safety_net_strictness == "warn"


def test_document_unreadable_action_defaults_to_reject():
    s = Settings()
    assert s.document_unreadable_action == "reject"


def test_anonymization_prompt_path_default():
    s = Settings()
    assert s.anonymization_prompt_path == "prompts/anonymize.txt"


def test_prompt_template_renders():
    template = Path("prompts/anonymize.txt").read_text()
    rendered = (
        template
        .replace("{existing_mapping}", '{"Microsoft": "SOFTWARE_COMPANY_01"}')
        .replace("{message_text}", "John Smith works at Microsoft.")
    )
    assert "John Smith works at Microsoft." in rendered
    assert '"Microsoft": "SOFTWARE_COMPANY_01"' in rendered
    assert "{existing_mapping}" not in rendered
    assert "{message_text}" not in rendered


def test_substitution_prompt_template_is_user_editable():
    template = Path("prompts/substitute.md").read_text()

    assert "{entities}" in template
    assert "{query}" in template
    assert "sensitivity" in template
    assert "needs_clarification" in template


def test_docker_image_includes_user_editable_prompts():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "COPY prompts/ ./prompts/" in dockerfile


def test_docker_compose_uses_settings_environment_names():
    compose = (ROOT / "docker-compose.yml").read_text()

    assert "REDIS_URL=redis://redis:6379/0" in compose
    assert "API_KEY=sk-heyjude-dev" in compose
    assert "LOCAL_LLM_URL=" in compose
    assert "LOCAL_LLM_MODEL=qwen3.5:4b" in compose
    assert "EXTERNAL_LLM_MODEL=ollama_chat/qwen3.5:4b" in compose
    assert "EXTERNAL_LLM_API_BASE=http://host.docker.internal:11434" in compose
    assert "HEY_JUDE_" not in compose


def test_gateway_port_is_4005_in_runtime_files():
    compose = (ROOT / "docker-compose.yml").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()
    readme = (ROOT / "README.md").read_text()

    assert '"4005:4005"' in compose
    assert "EXPOSE 4005" in dockerfile
    assert '"4005"' in dockerfile
    assert "localhost:4005" in readme
    assert "localhost:8000" not in readme
    assert '"8000:8000"' not in compose
    assert "EXPOSE 8000" not in dockerfile


def test_readme_documents_actual_environment_names():
    readme = (ROOT / "README.md").read_text()

    assert "`REDIS_URL`" in readme
    assert "`API_KEY`" in readme
    assert "`LOCAL_LLM_URL`" in readme
    assert "`LOCAL_LLM_MODEL`" in readme
    assert "`EXTERNAL_LLM_MODEL`" in readme
    assert "`EXTERNAL_LLM_API_BASE`" in readme
    assert "`DOCUMENT_UNREADABLE_ACTION`" in readme
    assert "`HEY_JUDE_" not in readme
