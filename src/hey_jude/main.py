from contextlib import asynccontextmanager

from fastapi import FastAPI

from hey_jude.config import Settings, settings as default_settings
from hey_jude.redis_client import RedisClient
from hey_jude.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis_client = RedisClient(app.state.settings.redis_url)
    await app.state.redis_client.connect()
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
