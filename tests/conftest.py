import json

import httpx
import pytest
import respx

from app.config import settings


class MockInferenceServer:
    """
    Simulates the external inference endpoint as a test double.

    Pass rate_limit_every=N to 429 on every Nth call.
    Pass retry_after=0 to skip real sleeps in backoff tests.
    """

    def __init__(self, rate_limit_every: int = 0, retry_after: int = 1):
        self._rate_limit_every = rate_limit_every
        self._retry_after = retry_after
        self._call_count = 0
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self._call_count += 1
        payload = json.loads(request.read())
        self.requests.append(payload)

        if self._rate_limit_every and self._call_count % self._rate_limit_every == 0:
            return httpx.Response(
                429,
                json={"detail": "Too Many Requests"},
                headers={"Retry-After": str(self._retry_after)},
            )

        return httpx.Response(
            200,
            json={
                "prompt_id": payload.get("prompt_id", ""),
                "result": f"mock result for: {payload.get('prompt', '')}",
            },
        )


@pytest.fixture
def mock_inference():
    """Always-200 inference endpoint. Returns the server so tests can inspect calls."""
    server = MockInferenceServer()
    with respx.mock() as mock:
        mock.post(settings.inference_url).mock(side_effect=server.handler)
        yield server


@pytest.fixture
def mock_inference_with_429():
    """
    429 on every 3rd call, with retry_after=0 so tests don't actually sleep.
    Returns the server so tests can assert retry counts.
    """
    server = MockInferenceServer(rate_limit_every=3, retry_after=0)
    with respx.mock() as mock:
        mock.post(settings.inference_url).mock(side_effect=server.handler)
        yield server
