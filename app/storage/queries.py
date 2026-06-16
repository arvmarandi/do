from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

async def create_batch(db: aiosqlite.Connection, prompts: list[str]) -> str:
    batch_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO batches (id, status, total, created_at) VALUES (?, 'pending', ?, ?)",
        (batch_id, len(prompts), _now()),
    )
    await db.commit()
    return batch_id


async def get_batch(db: aiosqlite.Connection, batch_id: str) -> dict[str, Any] | None:
    async with db.execute(
        "SELECT * FROM batches WHERE id = ?", (batch_id,)
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def set_batch_processing(db: aiosqlite.Connection, batch_id: str) -> None:
    await db.execute(
        "UPDATE batches SET status = 'processing' WHERE id = ?", (batch_id,)
    )
    await db.commit()


async def mark_batch_completed(db: aiosqlite.Connection, batch_id: str) -> None:
    await db.execute(
        """UPDATE batches
           SET status = 'completed', completed_at = ?
           WHERE id = ?""",
        (_now(), batch_id),
    )
    await db.commit()


async def increment_batch_counters(
    db: aiosqlite.Connection,
    batch_id: str,
    *,
    success: bool,
) -> None:
    if success:
        await db.execute(
            "UPDATE batches SET processed = processed + 1 WHERE id = ?", (batch_id,)
        )
    else:
        await db.execute(
            "UPDATE batches SET processed = processed + 1, failed = failed + 1 WHERE id = ?",
            (batch_id,),
        )
    await db.commit()


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

async def save_result(
    db: aiosqlite.Connection,
    batch_id: str,
    prompt_id: str,
    prompt: str,
    result: str | None,
    status: str,
    error: str | None = None,
) -> None:
    await db.execute(
        """INSERT INTO results (id, batch_id, prompt_id, prompt, result, status, error, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (str(uuid.uuid4()), batch_id, prompt_id, prompt, result, status, error, _now()),
    )
    await db.commit()


async def get_results(
    db: aiosqlite.Connection,
    batch_id: str,
    offset: int = 0,
    limit: int = 100,
) -> list[dict[str, Any]]:
    async with db.execute(
        "SELECT * FROM results WHERE batch_id = ? ORDER BY created_at LIMIT ? OFFSET ?",
        (batch_id, limit, offset),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]
