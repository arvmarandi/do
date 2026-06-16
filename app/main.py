from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: initialise DB connection pool here
    yield
    # TODO: close DB connection pool here


app = FastAPI(title="Batch Inference API", version="0.1.0", lifespan=lifespan)
app.include_router(router)
