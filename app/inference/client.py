from __future__ import annotations

import httpx


class RateLimitExhausted(Exception):
    """Raised when the inference endpoint returns HTTP 429."""
    def __init__(self, prompt_id: str, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit hit for prompt {prompt_id!r}; retry after {retry_after}s."
        )


class InferenceError(Exception):
    """Raised when the inference endpoint returns an unexpected error."""


class InferenceClient:
    """
    Async client for the external inference endpoint.

    Makes a single HTTP call per invocation. Retry/re-queue decisions
    are handled by the caller (the worker) so all retry logic lives in
    one place.

    Accepts an optional httpx.AsyncClient so tests can inject a mocked
    transport without patching globals.
    """

    def __init__(
        self,
        url: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def call(self, prompt_id: str, prompt: str) -> dict:
        """
        POST a single prompt to the inference endpoint.

        Returns the parsed JSON response on success.
        Raises RateLimitExhausted (with retry_after seconds) on HTTP 429.
        Raises InferenceError on any other non-200 status.
        """
        response = await self._client.post(
            self._url,
            json={"prompt_id": prompt_id, "prompt": prompt},
        )

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            delay = float(response.headers.get("Retry-After", 1.0))
            raise RateLimitExhausted(prompt_id, retry_after=delay)

        raise InferenceError(
            f"Inference endpoint returned {response.status_code} "
            f"for prompt {prompt_id!r}: {response.text}"
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> InferenceClient:
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()
