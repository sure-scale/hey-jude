from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    assert "`HEY_JUDE_" not in readme
