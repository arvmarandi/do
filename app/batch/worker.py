from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import aiosqlite

from app.inference.client import InferenceClient, RateLimitExhausted
from app.storage import queries


@dataclass
class Job:
    prompt_id: str
    prompt: str
    attempts: int = field(default=0)


async def run_worker(
    queue: asyncio.Queue[Job],
    db: aiosqlite.Connection,
    client: InferenceClient,
    batch_id: str,
    max_attempts: int,
) -> None:
    """
    Runs until cancelled. Pulls jobs from the queue one at a time and
    calls the inference endpoint.

    On HTTP 429: sleeps for Retry-After seconds then re-queues the job
    with attempts+1. Once a job reaches max_attempts it is recorded as
    failed rather than re-queued, so the batch always terminates.

    All other exceptions are recorded as failed immediately.
    """
    while True:
        job = await queue.get()
        try:
            data = await client.call(job.prompt_id, job.prompt)
            await queries.save_result(
                db, batch_id, job.prompt_id, job.prompt,
                result=data.get("result"), status="success",
            )
            await queries.increment_batch_counters(db, batch_id, success=True)

        except RateLimitExhausted as exc:
            if job.attempts < max_attempts:
                await asyncio.sleep(exc.retry_after)
                await queue.put(Job(job.prompt_id, job.prompt, job.attempts + 1))
            else:
                await queries.save_result(
                    db, batch_id, job.prompt_id, job.prompt,
                    result=None, status="failed", error=str(exc),
                )
                await queries.increment_batch_counters(db, batch_id, success=False)

        except Exception as exc:
            await queries.save_result(
                db, batch_id, job.prompt_id, job.prompt,
                result=None, status="failed", error=str(exc),
            )
            await queries.increment_batch_counters(db, batch_id, success=False)

        finally:
            queue.task_done()
