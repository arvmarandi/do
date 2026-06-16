import asyncio
import json

from fastapi import APIRouter, HTTPException, Request, UploadFile, File

from app.batch.processor import process_batch
from app.config import settings
from app.schemas import BatchAck, BatchRequest, BatchResults, BatchStatus
from app.storage import queries

router = APIRouter(prefix="/api/v1")


@router.post("/batches", status_code=202, response_model=BatchAck)
async def submit_batch(body: BatchRequest, request: Request):
    if len(body.prompts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch exceeds maximum size of {settings.max_batch_size} prompts.",
        )
    db = request.app.state.db
    batch_id = await queries.create_batch(db, body.prompts)
    asyncio.create_task(process_batch(batch_id, body.prompts, db))
    return BatchAck(batch_id=batch_id)


@router.post("/batches/upload", status_code=202, response_model=BatchAck)
async def upload_batch(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    try:
        prompts = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="File must contain a JSON array of strings.")
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts): # prompts must be a list and the prompts themselves must be strings
        raise HTTPException(status_code=400, detail="JSON must be an array of strings.")
    if len(prompts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch exceeds maximum size of {settings.max_batch_size} prompts.",
        )

    db = request.app.state.db
    batch_id = await queries.create_batch(db, prompts)
    asyncio.create_task(process_batch(batch_id, prompts, db))
    return BatchAck(batch_id=batch_id)


@router.get("/batches/{batch_id}", response_model=BatchStatus)
async def get_batch_status(batch_id: str, request: Request):
    db = request.app.state.db
    batch = await queries.get_batch(db, batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found.")
    return batch


@router.get("/batches/{batch_id}/results", response_model=BatchResults)
async def get_batch_results(batch_id: str, request: Request, offset: int = 0, limit: int = 100):
    db = request.app.state.db
    batch = await queries.get_batch(db, batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found.")
    results = await queries.get_results(db, batch_id, offset=offset, limit=limit)
    return BatchResults(batch_id=batch_id, offset=offset, limit=limit, results=results)
