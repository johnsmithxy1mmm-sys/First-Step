"""HTTP wrapper: httpx with exponential backoff and jitter."""

from __future__ import annotations

import logging
import random
import time

import httpx

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def make_client(timeout_sec: float = 30.0) -> httpx.Client:
    return httpx.Client(timeout=timeout_sec, headers={"User-Agent": "polymarket-longshot-bot/1.0"})


def get_with_backoff(
    client: httpx.Client,
    url: str,
    *,
    params: dict | None = None,
    max_retries: int = 4,
    base_delay: float = 1.0,
) -> httpx.Response:
    """GET retrying network errors and 429/5xx. Raises once retries are exhausted."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.get(url, params=params)
            if resp.status_code not in RETRYABLE_STATUS:
                resp.raise_for_status()
                return resp
            last_exc = httpx.HTTPStatusError(
                f"status {resp.status_code}", request=resp.request, response=resp
            )
            retry_after = resp.headers.get("retry-after")
            delay = float(retry_after) if retry_after else None
        except httpx.HTTPStatusError:
            raise  # non-retryable 4xx
        except httpx.HTTPError as exc:
            last_exc = exc
            delay = None
        if attempt >= max_retries:
            break
        delay = delay if delay is not None else base_delay * (2**attempt) + random.uniform(0, 0.5)
        log.warning("retry %s/%s for %s in %.1fs (%s)", attempt + 1, max_retries, url, delay, last_exc)
        time.sleep(delay)
    assert last_exc is not None
    raise last_exc
