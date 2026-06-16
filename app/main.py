from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.storage.database import get_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    app.state.db = await get_db()
    yield
    await app.state.db.close()


app = FastAPI(title="Batch Inference API", version="0.1.0", lifespan=lifespan)
app.include_router(router)
