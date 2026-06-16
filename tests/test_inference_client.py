import pytest
import respx
import httpx

from app.config import settings
from app.inference.client import InferenceClient, RateLimitExhausted, InferenceError


def make_client(**kwargs) -> InferenceClient:
    return InferenceClient(url=settings.inference_url, **kwargs)


@pytest.mark.asyncio
async def test_successful_call(mock_inference):
    async with make_client() as client:
        result = await client.call_with_retry("p1", "hello world")

    assert result["prompt_id"] == "p1"
    assert "mock result for: hello world" in result["result"]
    assert len(mock_inference.requests) == 1


@pytest.mark.asyncio
async def test_429_triggers_retry(monkeypatch):
    """A 429 should not fail the call — it should retry and eventually succeed."""
    async def fake_sleep(_): pass
    monkeypatch.setattr("app.inference.client.asyncio.sleep", fake_sleep)

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"prompt_id": "p1", "result": "ok"})

    with respx.mock() as mock:
        mock.post(settings.inference_url).mock(side_effect=handler)
        async with make_client(max_retries=3) as client:
            result = await client.call_with_retry("p1", "test prompt")

    assert result["prompt_id"] == "p1"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_after_header_is_respected(monkeypatch):
    """The sleep duration should come from the Retry-After header, not the default backoff."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("app.inference.client.asyncio.sleep", fake_sleep)

    with respx.mock() as mock:
        mock.post(settings.inference_url).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(200, json={"prompt_id": "p1", "result": "ok"}),
            ]
        )
        async with make_client(max_retries=3) as client:
            await client.call_with_retry("p1", "test")

    assert slept == [7.0]


@pytest.mark.asyncio
async def test_exponential_backoff_without_retry_after_header(monkeypatch):
    """Without a Retry-After header, sleep should follow 2^attempt backoff."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("app.inference.client.asyncio.sleep", fake_sleep)

    with respx.mock() as mock:
        mock.post(settings.inference_url).mock(
            side_effect=[
                httpx.Response(429),  # attempt 0 → sleep 1s (2^0)
                httpx.Response(429),  # attempt 1 → sleep 2s (2^1)
                httpx.Response(200, json={"prompt_id": "p1", "result": "ok"}),
            ]
        )
        async with make_client(max_retries=3) as client:
            await client.call_with_retry("p1", "test")

    assert slept == [1.0, 2.0]


@pytest.mark.asyncio
async def test_rate_limit_exhausted_raises(monkeypatch):
    """Exhausting all retries on 429 must raise RateLimitExhausted, not silently fail."""
    async def fake_sleep(_): pass
    monkeypatch.setattr("app.inference.client.asyncio.sleep", fake_sleep)

    with respx.mock() as mock:
        mock.post(settings.inference_url).mock(
            return_value=httpx.Response(429, headers={"Retry-After": "0"})
        )
        async with make_client(max_retries=2) as client:
            with pytest.raises(RateLimitExhausted):
                await client.call_with_retry("p1", "test")


@pytest.mark.asyncio
async def test_non_429_error_raises_immediately(monkeypatch):
    """A 500 should raise InferenceError immediately without retrying."""
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    monkeypatch.setattr("app.inference.client.asyncio.sleep", fake_sleep)

    with respx.mock() as mock:
        mock.post(settings.inference_url).mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        async with make_client(max_retries=3) as client:
            with pytest.raises(InferenceError):
                await client.call_with_retry("p1", "test")

    assert slept == [], "should not have slept — 500 is not retryable"
