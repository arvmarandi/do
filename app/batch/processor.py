from __future__ import annotations

import asyncio
import uuid

import aiosqlite

from app.batch.worker import Job, run_worker
from app.config import Settings, settings as default_settings
from app.inference.client import InferenceClient
from app.storage import queries


async def process_batch(
    batch_id: str,
    prompts: list[str],
    db: aiosqlite.Connection,
    cfg: Settings = default_settings,
    client: InferenceClient | None = None,
) -> None:
    """
    Entry point for background batch processing.

    Spawns exactly cfg.worker_pool_size coroutines — that number is the
    hard concurrency cap. queue.join() blocks until every prompt has been
    processed or recorded as failed, guaranteeing no prompt is silently dropped.
    """
    await queries.set_batch_processing(db, batch_id)

    queue: asyncio.Queue[Job] = asyncio.Queue() # create a coroutine queue for the to-be-processed prompts
    for _, prompt in enumerate(prompts):
        await queue.put(Job(prompt_id=str(uuid.uuid4()), prompt=prompt))

    owns_client = client is None
    if owns_client:
        client = InferenceClient(url=cfg.inference_url, max_retries=cfg.max_retries)

    worker_tasks = [
        asyncio.create_task(run_worker(queue, db, client, batch_id))
        for _ in range(cfg.worker_pool_size)
    ]

    try:
        await queue.join()
    finally:
        for task in worker_tasks:
            task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        if owns_client:
            await client.aclose()

    await queries.mark_batch_completed(db, batch_id)
