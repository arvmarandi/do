import httpx
import pytest
import respx

from app.config import settings
from app.inference.client import InferenceClient, InferenceError, RateLimitExhausted


def make_client(**kwargs) -> InferenceClient:
    return InferenceClient(url=settings.inference_url, **kwargs)


@pytest.mark.asyncio
async def test_successful_call(mock_inference):
    async with make_client() as client:
        result = await client.call("p1", "hello world")

    assert result["prompt_id"] == "p1"
    assert "mock result for: hello world" in result["result"]


@pytest.mark.asyncio
async def test_429_raises_rate_limit_exhausted():
    with respx.mock() as mock:
        mock.post(settings.inference_url).mock(
            return_value=httpx.Response(429, headers={"Retry-After": "5"})
        )
        async with make_client() as client:
            with pytest.raises(RateLimitExhausted) as exc_info:
                await client.call("p1", "test")

    assert exc_info.value.retry_after == 5.0


@pytest.mark.asyncio
async def test_429_retry_after_defaults_to_one_second():
    """When the Retry-After header is absent, retry_after should default to 1.0."""
    with respx.mock() as mock:
        mock.post(settings.inference_url).mock(
            return_value=httpx.Response(429)
        )
        async with make_client() as client:
            with pytest.raises(RateLimitExhausted) as exc_info:
                await client.call("p1", "test")

    assert exc_info.value.retry_after == 1.0


@pytest.mark.asyncio
async def test_non_429_error_raises_inference_error():
    """A 500 should raise InferenceError immediately."""
    with respx.mock() as mock:
        mock.post(settings.inference_url).mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        async with make_client() as client:
            with pytest.raises(InferenceError):
                await client.call("p1", "test")
