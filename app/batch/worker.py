from __future__ import annotations

import asyncio
from dataclasses import dataclass

import aiosqlite

from app.inference.client import InferenceClient
from app.storage import queries


@dataclass
class Job:
    prompt_id: str
    prompt: str


async def run_worker(
    queue: asyncio.Queue[Job],
    db: aiosqlite.Connection,
    client: InferenceClient,
    batch_id: str,
) -> None:
    """
    Runs until cancelled. Pulls jobs from the queue one at a time,
    calls the inference endpoint, and writes the result to the DB.
    Failures are recorded per-prompt rather than propagated, so one
    bad prompt never kills the worker or stalls the batch.
    """
    while True:
        job = await queue.get()
        try:
            data = await client.call_with_retry(job.prompt_id, job.prompt)
            await queries.save_result(
                db, batch_id, job.prompt_id, job.prompt,
                result=data.get("result"), status="success",
            )
            await queries.increment_batch_counters(db, batch_id, success=True)
        except Exception as exc:
            await queries.save_result(
                db, batch_id, job.prompt_id, job.prompt,
                result=None, status="failed", error=str(exc),
            )
            await queries.increment_batch_counters(db, batch_id, success=False)
        finally:
            queue.task_done()
