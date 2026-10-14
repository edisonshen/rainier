"""Anthropic Message Batches transport for thesis calls (50% price, async).

``ThesisLLMBatcher`` lets concurrently running ``generate_thesis`` tasks submit
their LLM calls as one batch: each call parks on a future, and once every live
task is either parked or finished the parked calls go out together. Retries
form the next round. A batch that fails or outlasts its timeout is cancelled
and its calls fall back to the synchronous path at full price.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple

import httpx

log = logging.getLogger(__name__)

ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
BATCH_PRICE_FACTOR = 0.5

_NATIVE_PARAM_KEYS = ("max_tokens", "thinking", "output_config", "temperature")


class LLMReply(NamedTuple):
    text: str
    prompt_tokens: int
    completion_tokens: int
    price_factor: float = 1.0


class BatchTimeoutError(TimeoutError):
    pass


def to_anthropic_params(completion_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate LiteLLM completion kwargs into a native Messages API body."""
    model = str(completion_kwargs["model"]).removeprefix("anthropic/")
    params: dict[str, Any] = {"model": model}
    messages: list[dict[str, Any]] = []
    for message in completion_kwargs["messages"]:
        if message["role"] == "system":
            params["system"] = message["content"]
            continue
        messages.append({"role": message["role"], "content": _native_content(message["content"])})
    params["messages"] = messages
    for key in _NATIVE_PARAM_KEYS:
        if key in completion_kwargs:
            params[key] = completion_kwargs[key]
    return params


def _native_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        if part.get("type") == "image_url":
            header, data = part["image_url"]["url"].split(",", 1)
            media_type = header.removeprefix("data:").split(";", 1)[0]
            parts.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            })
        else:
            parts.append(part)
    return parts


def _cancel_batch(batch_id: str, headers: dict[str, str]) -> None:
    try:
        httpx.post(f"{ANTHROPIC_API_BASE}/messages/batches/{batch_id}/cancel",
                   headers=headers, timeout=30)
    except Exception:
        log.warning("thesis_batch_cancel_failed batch_id=%s", batch_id, exc_info=True)


def _fetch_results(
    url: str, headers: dict[str, str], *, retry_wait_s: float, attempts: int = 3
) -> httpx.Response:
    for attempt in range(1, attempts + 1):
        try:
            resp = httpx.get(url, headers=headers, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            return resp
        except Exception:
            if attempt == attempts:
                raise
            log.warning("thesis_batch_results_retry attempt=%d url=%s", attempt, url,
                        exc_info=True)
            time.sleep(retry_wait_s)
    raise AssertionError("unreachable")


def run_message_batch(
    requests: list[dict[str, Any]],
    *,
    api_key: str,
    timeout_s: float,
    poll_s: float,
) -> dict[int, LLMReply | str]:
    """Submit one batch, wait for it, return ``{request_index: reply | error}``.

    Raises ``BatchTimeoutError`` (after cancelling) when the batch has not ended
    within ``timeout_s``.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body = {"requests": [{"custom_id": f"r{i}", "params": p} for i, p in enumerate(requests)]}
    resp = httpx.post(f"{ANTHROPIC_API_BASE}/messages/batches", headers=headers, json=body,
                      timeout=60)
    resp.raise_for_status()
    batch = resp.json()
    log.info("thesis_batch_submitted batch_id=%s requests=%d", batch["id"], len(requests))

    batch_id = batch["id"]
    try:
        deadline = time.monotonic() + timeout_s
        while batch["processing_status"] != "ended":
            if time.monotonic() >= deadline:
                raise BatchTimeoutError(f"batch {batch_id} not done after {timeout_s:.0f}s")
            time.sleep(poll_s)
            resp = httpx.get(f"{ANTHROPIC_API_BASE}/messages/batches/{batch_id}",
                             headers=headers, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
    except Exception:
        # The caller falls back to sync calls; cancel so the batch can't also bill.
        _cancel_batch(batch_id, headers)
        raise
    # The batch has ended and is already billed, so retry the download rather
    # than paying again through the sync fallback.
    resp = _fetch_results(batch["results_url"], headers, retry_wait_s=poll_s)
    results: dict[int, LLMReply | str] = {}
    for line in resp.text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index = int(str(row["custom_id"]).removeprefix("r"))
        result = row["result"]
        if result.get("type") != "succeeded":
            results[index] = f"batch request {result.get('type')}: {result.get('error')}"
            continue
        message = result["message"]
        usage = message.get("usage") or {}
        text = "".join(b.get("text", "") for b in message.get("content", [])
                       if b.get("type") == "text")
        results[index] = LLMReply(
            text,
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
            BATCH_PRICE_FACTOR,
        )
    log.info("thesis_batch_done batch_id=%s counts=%s", batch["id"], batch.get("request_counts"))
    return results


class ThesisLLMBatcher:
    """Collects thesis LLM calls from concurrent tasks into Message Batches."""

    def __init__(
        self,
        *,
        api_key: str,
        sync_call: Callable[[dict[str, Any]], tuple[str, int, int]],
        timeout_s: float,
        poll_s: float,
    ) -> None:
        self._api_key = api_key
        self._sync_call = sync_call
        self._timeout_s = timeout_s
        self._poll_s = poll_s
        self._live = 0
        self._parked: list[tuple[dict[str, Any], asyncio.Future[LLMReply]]] = []
        self._rounds: set[asyncio.Task[None]] = set()

    async def run_all(self, coros: list[Awaitable[Any]]) -> list[Any]:
        """Run ``coros`` concurrently; their ``call()``s are batched together."""
        self._live = len(coros)
        return await asyncio.gather(*(self._tracked(c) for c in coros), return_exceptions=True)

    async def call(self, completion_kwargs: dict[str, Any]) -> LLMReply:
        future: asyncio.Future[LLMReply] = asyncio.get_running_loop().create_future()
        self._parked.append((completion_kwargs, future))
        self._flush_when_all_parked()
        return await future

    async def _tracked(self, coro: Awaitable[Any]) -> Any:
        try:
            return await coro
        finally:
            self._live -= 1
            self._flush_when_all_parked()

    def _flush_when_all_parked(self) -> None:
        if not self._parked or len(self._parked) < self._live:
            return
        parked, self._parked = self._parked, []
        task = asyncio.ensure_future(self._run_round(parked))
        self._rounds.add(task)
        task.add_done_callback(self._rounds.discard)

    async def _run_round(
        self, parked: list[tuple[dict[str, Any], asyncio.Future[LLMReply]]]
    ) -> None:
        results: dict[int, LLMReply | str] = {}
        try:
            results = await asyncio.to_thread(
                run_message_batch,
                [to_anthropic_params(kwargs) for kwargs, _ in parked],
                api_key=self._api_key,
                timeout_s=self._timeout_s,
                poll_s=self._poll_s,
            )
        except Exception as exc:
            log.warning("thesis_batch_failed_falling_back_to_sync error=%s", exc)
        for index, (kwargs, future) in enumerate(parked):
            reply = results.get(index)
            if isinstance(reply, LLMReply):
                future.set_result(reply)
                continue
            if reply is not None:
                log.warning("thesis_batch_request_failed index=%d error=%s", index, reply)
            try:
                text, prompt_tokens, completion_tokens = await asyncio.to_thread(
                    self._sync_call, kwargs
                )
                future.set_result(LLMReply(text, prompt_tokens, completion_tokens))
            except Exception as exc:
                future.set_exception(exc)
