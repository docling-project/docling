"""Internal bounded async scheduling helpers for client-side batch work."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

import httpx

T_Item = TypeVar("T_Item")
T_Result = TypeVar("T_Result")


async def _run_bounded(
    items: list[T_Item],
    process_one: Callable[[int, T_Item, httpx.AsyncClient], Awaitable[T_Result]],
    async_client: httpx.AsyncClient,
    max_in_flight: int,
) -> AsyncIterator[tuple[int, T_Result | BaseException]]:
    """Yield item outcomes in completion order with bounded concurrency."""

    if not items:
        return

    worker_count = min(len(items), max(1, max_in_flight))
    queue: asyncio.Queue[tuple[int, T_Result | BaseException]] = asyncio.Queue()
    index_lock = asyncio.Lock()
    next_idx = 0

    async def worker() -> None:
        nonlocal next_idx
        while True:
            async with index_lock:
                if next_idx >= len(items):
                    return
                idx = next_idx
                next_idx += 1

            item = items[idx]
            try:
                result = await process_one(idx, item, async_client)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                await queue.put((idx, exc))
            else:
                await queue.put((idx, result))

    tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]
    completed = 0
    try:
        while completed < len(items):
            yield await queue.get()
            completed += 1
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
