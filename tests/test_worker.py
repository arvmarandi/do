from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from app.batch.processor import process_batch
from app.config import Settings
from app.inference.client import InferenceClient, RateLimitExhausted
from app.storage.database import init_db
from app.storage import queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def make_db(tmp_path) -> aiosqlite.Connection:
    db_path = str(tmp_path / "test.db")
    from app.storage import database
    original = database.settings
    database.settings = Settings(db_path=db_path)
    await init_db()
    database.settings = original

    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


@pytest.fixture
async def db(tmp_path):
    conn = await make_db(tmp_path)
    yield conn
    await conn.close()


def make_settings(**kwargs) -> Settings:
    return Settings(
        inference_url="http://test-inference/infer",
        db_path=":memory:",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fake clients
# ---------------------------------------------------------------------------

class FakeClient(InferenceClient):
    """Instant-return client that tracks peak concurrency."""

    def __init__(self, peak_tracker: list[int], delay: float = 0.01):
        self._url = "http://fake"
        self._owns_client = False
        self._peak_tracker = peak_tracker
        self._delay = delay
        self._active = 0

    async def call(self, prompt_id: str, prompt: str) -> dict:
        self._active += 1
        self._peak_tracker.append(self._active)
        await asyncio.sleep(self._delay)
        self._active -= 1
        return {"prompt_id": prompt_id, "result": f"ok: {prompt}"}

    async def aclose(self) -> None:
        pass


class AlwaysFailClient(InferenceClient):
    def __init__(self):
        self._url = "http://fake"
        self._owns_client = False

    async def call(self, prompt_id: str, prompt: str) -> dict:
        raise RuntimeError("inference failed")

    async def aclose(self) -> None:
        pass


class RateLimitedClient(InferenceClient):
    """
    Returns 429 for the first `fail_times` calls per prompt, then succeeds.
    Tracks total call count to let tests assert re-queue behaviour.
    """

    def __init__(self, fail_times: int = 1, retry_after: float = 0.0):
        self._url = "http://fake"
        self._owns_client = False
        self._fail_times = fail_times
        self._retry_after = retry_after
        self._call_counts: dict[str, int] = {}
        self.total_calls: int = 0

    async def call(self, prompt_id: str, prompt: str) -> dict:
        self.total_calls += 1
        count = self._call_counts.get(prompt_id, 0)
        self._call_counts[prompt_id] = count + 1
        if count < self._fail_times:
            raise RateLimitExhausted(prompt_id, retry_after=self._retry_after)
        return {"prompt_id": prompt_id, "result": f"ok: {prompt}"}

    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_prompts_processed(db):
    prompts = [f"prompt {i}" for i in range(20)]
    batch_id = await queries.create_batch(db, prompts)

    cfg = make_settings(worker_pool_size=5)
    await process_batch(batch_id, prompts, db, cfg=cfg, client=FakeClient([]))

    batch = await queries.get_batch(db, batch_id)
    assert batch["status"] == "completed"
    assert batch["processed"] == 20
    assert batch["failed"] == 0

    results = await queries.get_results(db, batch_id, limit=20)
    assert len(results) == 20
    assert all(r["status"] == "success" for r in results)


@pytest.mark.asyncio
async def test_worker_pool_size_caps_concurrency(db):
    """Peak active workers must never exceed worker_pool_size."""
    peak_tracker: list[int] = []
    prompts = [f"prompt {i}" for i in range(30)]
    batch_id = await queries.create_batch(db, prompts)

    pool_size = 5
    cfg = make_settings(worker_pool_size=pool_size)
    client = FakeClient(peak_tracker, delay=0.02)
    await process_batch(batch_id, prompts, db, cfg=cfg, client=client)

    assert max(peak_tracker) <= pool_size


@pytest.mark.asyncio
async def test_failed_prompts_recorded_batch_still_completes(db):
    """A prompt that raises must be saved as failed; the batch must still complete."""
    prompts = [f"prompt {i}" for i in range(5)]
    batch_id = await queries.create_batch(db, prompts)

    cfg = make_settings(worker_pool_size=3)
    await process_batch(batch_id, prompts, db, cfg=cfg, client=AlwaysFailClient())

    batch = await queries.get_batch(db, batch_id)
    assert batch["status"] == "completed"
    assert batch["processed"] == 5
    assert batch["failed"] == 5

    results = await queries.get_results(db, batch_id, limit=10)
    assert all(r["status"] == "failed" for r in results)
    assert all(r["error"] is not None for r in results)


@pytest.mark.asyncio
async def test_429_prompt_is_requeued_and_eventually_succeeds(db):
    """A prompt that 429s once should be re-queued and succeed on the retry."""
    prompts = ["prompt 0"]
    batch_id = await queries.create_batch(db, prompts)

    cfg = make_settings(worker_pool_size=2, max_retries=3)
    client = RateLimitedClient(fail_times=1, retry_after=0.0)
    await process_batch(batch_id, prompts, db, cfg=cfg, client=client)

    assert client.total_calls == 2

    batch = await queries.get_batch(db, batch_id)
    assert batch["status"] == "completed"
    assert batch["failed"] == 0

    results = await queries.get_results(db, batch_id, limit=5)
    assert results[0]["status"] == "success"


@pytest.mark.asyncio
async def test_429_prompt_fails_after_max_retries(db):
    """A prompt that 429s on every attempt must be recorded as failed after max_retries."""
    prompts = ["prompt 0"]
    batch_id = await queries.create_batch(db, prompts)

    max_retries = 2
    cfg = make_settings(worker_pool_size=2, max_retries=max_retries)
    client = RateLimitedClient(fail_times=999, retry_after=0.0)
    await process_batch(batch_id, prompts, db, cfg=cfg, client=client)

    # initial attempt + max_retries re-queues
    assert client.total_calls == max_retries + 1

    batch = await queries.get_batch(db, batch_id)
    assert batch["status"] == "completed"
    assert batch["failed"] == 1

    results = await queries.get_results(db, batch_id, limit=5)
    assert results[0]["status"] == "failed"
