# Batch Inference API

A REST API that accepts batches of AI prompts, processes them concurrently against an inference endpoint, handles rate limiting with automatic retry and re-queue logic, and persists results to SQLite.

---

## Architecture

```
Client
  │
  └─► POST /api/v1/batches
        │
        ├─► Validate → write batch row to SQLite (status=pending)
        ├─► Return 202 Accepted  {"batch_id": "<uuid>"}
        │
        └─► asyncio.create_task(process_batch)
                │
                ├─► Single asyncio.Queue  (all prompts loaded upfront)
                │
                ├─► Spawn WORKER_POOL_SIZE coroutines
                │       each: queue.get() → client.call() → save result
                │
                ├─► On HTTP 429:
                │       sleep(Retry-After seconds)
                │       re-queue job with attempts + 1
                │       fail permanently once attempts == MAX_RETRIES
                │
                └─► queue.join() → mark batch completed in SQLite
```

### Concurrency model

- One `asyncio.Queue` is created per batch. All workers for that batch share the same queue — there is no per-worker queue.
- The worker pool size is the hard concurrency cap. Spawning exactly `WORKER_POOL_SIZE` coroutines means at most that many prompts are actively calling the inference endpoint at any moment.
- The queue is unbounded because all prompts are already in memory (received as a JSON request body). Bounding the queue to pool size would risk deadlock when re-queuing after a 429.
- Each batch gets its own isolated queue and worker pool. Concurrent batches do not share workers.
- `await asyncio.sleep()` on a 429 suspends only the one coroutine that hit the rate limit — all other workers continue processing.

---

## Project structure

```
app/
├── main.py               # FastAPI app factory and lifespan (DB init)
├── config.py             # Settings via pydantic-settings (.env support)
├── schemas.py            # Pydantic request/response models
├── api/
│   └── routes.py         # POST /batches, POST /batches/upload,
│                         # GET /batches/{id}, GET /batches/{id}/results
├── batch/
│   ├── processor.py      # process_batch(): fills queue, spawns workers
│   └── worker.py         # run_worker(): per-prompt call, retry, re-queue
├── inference/
│   └── client.py         # InferenceClient: single HTTP call, raises on 429
└── storage/
    ├── database.py       # SQLite init and connection factory
    └── queries.py        # Typed async query helpers

mock_server/
└── main.py               # Standalone mock inference server (demo only)

demo/
└── prompts.json          # Sample batch of 20 prompts for demo

tests/
├── conftest.py                # MockInferenceServer test double (respx)
├── test_inference_client.py   # Client: 429 raises, Retry-After, error handling
└── test_worker.py             # Worker pool: concurrency cap, re-queue, failure recording
```

---

## Setup

### Requirements

- Python 3.12+

### Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `WORKER_POOL_SIZE` | `10` | Max concurrent workers per batch |
| `MAX_RETRIES` | `3` | Max re-queue attempts per prompt on 429 |
| `INFERENCE_URL` | `http://localhost:8081/infer` | Inference endpoint URL |
| `DB_PATH` | `batches.db` | SQLite database file path |

---

## Running

### 1. Start the mock inference server

In one terminal:

```bash
uvicorn mock_server.main:app --port 8081
```

Control 429 behaviour with env vars:

```bash
RATE_LIMIT_EVERY=3 uvicorn mock_server.main:app --port 8081   # 429 every 3rd request
RATE_LIMIT_EVERY=0 uvicorn mock_server.main:app --port 8081   # no 429s
```

### 2. Start the API

In a second terminal:

```bash
uvicorn app.main:app --port 8000 --reload
```

---

## API endpoints

### Submit a batch (JSON body)

```bash
curl -X POST http://localhost:8000/api/v1/batches \
  -H "Content-Type: application/json" \
  -d @demo/prompts.json
```

### Submit a batch (file upload)

```bash
curl -X POST http://localhost:8000/api/v1/batches/upload \
  -F "file=@demo/prompts.json"
```

Both return `202 Accepted` immediately while processing continues in the background:

```json
{
  "batch_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "message": "Batch accepted and queued for processing."
}
```

### Check batch status

```bash
curl http://localhost:8000/api/v1/batches/{batch_id}
```

```json
{
  "id": "f47ac10b-...",
  "status": "processing",
  "total": 20,
  "processed": 14,
  "failed": 1,
  "created_at": "2026-06-16T10:00:00+00:00",
  "completed_at": null
}
```

`status` is one of: `pending` → `processing` → `completed`.

### Retrieve results

```bash
curl "http://localhost:8000/api/v1/batches/{batch_id}/results?offset=0&limit=100"
```

```json
{
  "batch_id": "f47ac10b-...",
  "offset": 0,
  "limit": 100,
  "results": [
    {
      "id": "...",
      "batch_id": "f47ac10b-...",
      "prompt_id": "...",
      "prompt": "Explain the concept of recursion in simple terms.",
      "result": "Inference result for: Explain the concept of recursion in simple terms.",
      "status": "success",
      "error": null,
      "created_at": "2026-06-16T10:00:01+00:00"
    }
  ]
}
```

---

## Design tradeoffs

### asyncio coroutines vs threads
Workers are asyncio coroutines, not OS threads. This is the right fit for I/O-bound work (waiting on HTTP responses) — coroutines are cheap to spawn and don't need locking for shared state. The tradeoff is that any accidentally blocking call (a slow synchronous library, a CPU-heavy operation) would stall the entire event loop. Everything in this service uses async-native libraries (`httpx`, `aiosqlite`) to avoid that.

### Single shared SQLite connection
One `aiosqlite.Connection` is opened at startup and shared across all requests via `app.state.db`. SQLite in WAL mode handles concurrent readers safely, and routing all writes through a single async connection avoids contention. The tradeoff is that this does not scale horizontally — running multiple server processes would have them fighting over the same file. For a single-process service this is the right default; swap for PostgreSQL if horizontal scaling becomes a requirement.

### Re-queue on 429 vs retrying inside the client
When a worker hits a 429, it sleeps for `Retry-After` seconds and then puts the job back on the shared queue rather than retrying in a tight loop inside the HTTP client. This means the sleeping worker releases the job for any free worker to pick up after the delay, other workers continue processing unaffected, and all retry state (`attempts`) lives in one place on the `Job` object. The tradeoff is that if many workers hit 429 simultaneously, several are sleeping at once, temporarily shrinking the effective pool size.

### Unbounded queue
The queue is unbounded because all prompts arrive in a single JSON request body and are already in memory. Bounding the queue to pool size would risk deadlock: if all workers are sleeping on a 429 and the queue is full, no one can drain it. If very large batches (100k+ prompts) become a requirement, the right move is a bounded queue combined with a streaming producer that reads prompts from a file line by line, creating natural backpressure without loading everything into memory.

### Per-batch worker pools
Each batch gets its own isolated queue and worker pool. This keeps the implementation simple and batches completely independent — a slow or rate-limited batch cannot starve another. The tradeoff is that concurrent batches each spin up `WORKER_POOL_SIZE` workers, so submitting 10 batches simultaneously creates 100 concurrent workers all hitting the inference endpoint. A global semaphore or batch queue at the `process_batch` level would be needed to limit total concurrent workers across batches in a production setting.

### Prompts loaded into memory upfront
The entire prompt list is read from the request body before processing starts. This simplifies the implementation but means very large batches consume memory proportional to batch size for the lifetime of processing. For the scale described (1,000 items), this is not a concern.

---

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -v
```

### What is tested

| File | Coverage |
|---|---|
| `test_inference_client.py` | 429 raises `RateLimitExhausted` with correct `retry_after`; default backoff; non-429 errors fail immediately |
| `test_worker.py` | All prompts processed; peak concurrency ≤ pool size; failed prompts recorded without stopping batch; 429 re-queues and eventually succeeds; permanent failure after max retries |
