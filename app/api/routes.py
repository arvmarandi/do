from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1")


class BatchRequest(BaseModel):
    prompts: list[str]


class BatchAck(BaseModel):
    batch_id: str
    message: str = "Batch accepted and queued for processing."


@router.post("/batches", status_code=202, response_model=BatchAck)
async def submit_batch(body: BatchRequest):
    """Accept a JSON array of prompts and kick off background processing."""
    # TODO: persist batch + start background task
    return BatchAck(batch_id="placeholder-uuid")


@router.post("/batches/upload", status_code=202, response_model=BatchAck)
async def upload_batch(file: UploadFile = File(...)):
    """Accept a JSON file upload containing an array of prompts."""
    # TODO: parse file, persist batch + start background task
    return BatchAck(batch_id="placeholder-uuid")


@router.get("/batches/{batch_id}")
async def get_batch_status(batch_id: str):
    """Return processing status and progress counters for a batch."""
    # TODO: query DB
    raise HTTPException(status_code=404, detail="Batch not found")


@router.get("/batches/{batch_id}/results")
async def get_batch_results(batch_id: str, offset: int = 0, limit: int = 100):
    """Return paginated inference results for a completed batch."""
    # TODO: query DB
    raise HTTPException(status_code=404, detail="Batch not found")
