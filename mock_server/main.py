"""
Mock inference server for demo purposes.

Returns 200 with a fake result most of the time, and 429 every Nth request.
N is controlled by the RATE_LIMIT_EVERY env var (default: 5).
Set RATE_LIMIT_EVERY=0 to disable 429s entirely.
"""
import os
import time
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

RATE_LIMIT_EVERY = int(os.getenv("RATE_LIMIT_EVERY", 5))
RETRY_AFTER = int(os.getenv("RETRY_AFTER", 2))

app = FastAPI(title="Mock Inference Server")

_lock = Lock()
_call_count = 0


class InferenceRequest(BaseModel):
    prompt_id: str
    prompt: str


@app.post("/infer")
async def infer(body: InferenceRequest, request: Request):
    global _call_count

    with _lock:
        _call_count += 1
        count = _call_count

    if RATE_LIMIT_EVERY and count % RATE_LIMIT_EVERY == 0:
        print(f"  [mock] call #{count} → 429  (prompt_id={body.prompt_id!r})")
        return JSONResponse(
            status_code=429,
            content={"detail": "Too Many Requests"},
            headers={"Retry-After": str(RETRY_AFTER)},
        )

    # Simulate a small amount of processing time
    result = f"Inference result for: {body.prompt}"
    print(f"  [mock] call #{count} → 200  (prompt_id={body.prompt_id!r})")
    return {"prompt_id": body.prompt_id, "result": result}


@app.get("/health")
async def health():
    return {"status": "ok", "call_count": _call_count, "rate_limit_every": RATE_LIMIT_EVERY}
