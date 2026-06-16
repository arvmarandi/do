from __future__ import annotations

from pydantic import BaseModel


class BatchRequest(BaseModel):
    prompts: list[str]


class BatchAck(BaseModel):
    batch_id: str
    message: str = "Batch accepted and queued for processing."


class BatchStatus(BaseModel):
    id: str
    status: str
    total: int
    processed: int
    failed: int
    created_at: str
    completed_at: str | None


class ResultItem(BaseModel):
    id: str
    batch_id: str
    prompt_id: str
    prompt: str
    result: str | None
    status: str
    error: str | None
    created_at: str


class BatchResults(BaseModel):
    batch_id: str
    offset: int
    limit: int
    results: list[ResultItem]
