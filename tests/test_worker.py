from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from app.batch.processor import process_batch
from app.config import Settings
from app.inference.client import InferenceClient
from app.storage.database import init_db
from app.storage import queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def make_db(tmp_path) -> aiosqlite.Connection:
    db_path = str(tmp_path / "test.db")
    test_settings = Settings(db_path=db_path)
    await init_db.__wrapped__(test_settings) if hasattr(init_db, "__wrapped__") else None

    import aiosqlite as _aiosqlite
    from app.storage import database
    original = database.settings
    database.settings = test_settings
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
# Fake InferenceClient
# ---------------------------------------------------------------------------

class FakeClient(InferenceClient):
    """Instant-return client that tracks peak concurrency."""

    def __init__(self, peak_tracker: list[int], delay: float = 0.01):
        self._url = "http://fake"
        self._max_retries = 0
        self._owns_client = False
        self._peak_tracker = peak_tracker
        self._delay = delay
        self._active = 0

    async def call_with_retry(self, prompt_id: str, prompt: str) -> dict:
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
        self._max_retries = 0
        self._owns_client = False

    async def call_with_retry(self, prompt_id: str, prompt: str) -> dict:
        raise RuntimeError("inference failed")

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
