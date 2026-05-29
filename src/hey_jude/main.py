from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from hey_jude.config import Settings, settings as default_settings
from hey_jude.redis_client import RedisClient
from hey_jude.routes import router
from hey_jude.services.known_entities import load_known_entities


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis_client = RedisClient(app.state.settings.redis_url)
    await app.state.redis_client.connect()
    prompt_path = Path(app.state.settings.anonymization_prompt_path)
    if prompt_path.exists():
        app.state.anonymization_prompt = prompt_path.read_text()
    else:
        app.state.anonymization_prompt = None
    known_entities_path = app.state.settings.known_entities_path
    app.state.known_entities = (
        load_known_entities(known_entities_path) if known_entities_path else []
    )
    yield
    await app.state.redis_client.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="Hey Jude",
        description="Context-preserving pseudonymization gateway",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings or default_settings
    app.include_router(router)
    return app


app = create_app()
