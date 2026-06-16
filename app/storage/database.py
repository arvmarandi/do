import aiosqlite

from app.config import settings

_CREATE_BATCHES = """
CREATE TABLE IF NOT EXISTS batches (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'pending',
    total       INTEGER NOT NULL,
    processed   INTEGER NOT NULL DEFAULT 0,
    failed      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    completed_at TEXT
)
"""

_CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS results (
    id          TEXT PRIMARY KEY,
    batch_id    TEXT NOT NULL,
    prompt_id   TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    result      TEXT,
    status      TEXT NOT NULL,
    error       TEXT,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES batches(id)
)
"""

_CREATE_RESULTS_IDX = """
CREATE INDEX IF NOT EXISTS idx_results_batch_id ON results (batch_id)
"""


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(settings.db_path)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db() -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute(_CREATE_BATCHES)
        await db.execute(_CREATE_RESULTS)
        await db.execute(_CREATE_RESULTS_IDX)
        await db.commit()
