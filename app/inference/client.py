from __future__ import annotations

import asyncio

import httpx


class RateLimitExhausted(Exception):
    """Raised when a prompt exhausts all retry attempts due to HTTP 429."""


class InferenceError(Exception):
    """Raised when the inference endpoint returns an unexpected error."""


class InferenceClient:
    """
    Async client for the external inference endpoint.

    Accepts an optional httpx.AsyncClient so tests can inject a mocked transport
    without patching globals.
    """

    def __init__(
        self,
        url: str,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._max_retries = max_retries
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def call_with_retry(self, prompt_id: str, prompt: str) -> dict:
        """
        POST a single prompt to the inference endpoint.

        Retries on HTTP 429 using the Retry-After header value as the sleep
        duration, falling back to exponential backoff (1s, 2s, 4s, ...).
        Any other non-200 status raises InferenceError immediately.
        """
        for attempt in range(self._max_retries + 1):
            response = await self._client.post(
                self._url,
                json={"prompt_id": prompt_id, "prompt": prompt},
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                if attempt == self._max_retries:
                    raise RateLimitExhausted(
                        f"Prompt {prompt_id!r} hit rate limit on all "
                        f"{self._max_retries + 1} attempts."
                    )
                delay = float(response.headers.get("Retry-After", 2**attempt))
                await asyncio.sleep(delay)
                continue

            raise InferenceError(
                f"Inference endpoint returned {response.status_code} "
                f"for prompt {prompt_id!r}: {response.text}"
            )

        # unreachable, but keeps type checkers happy
        raise InferenceError("Unexpected exit from retry loop.")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> InferenceClient:
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()
