"""Internal bounded async scheduling helpers for client-side batch work."""

from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Sized,
)
from dataclasses import dataclass
from typing import TypeVar

import httpx

T_Item = TypeVar("T_Item")
T_Result = TypeVar("T_Result")


@dataclass(frozen=True, slots=True)
class _WorkerDone:
    pass


async def _run_bounded(
    items: Iterable[T_Item],
    process_one: Callable[[int, T_Item, httpx.AsyncClient], Awaitable[T_Result]],
    async_client: httpx.AsyncClient,
    max_in_flight: int,
) -> AsyncIterator[tuple[int, T_Item, T_Result | BaseException]]:
    """Yield item outcomes in completion order with bounded concurrency."""

    if isinstance(items, Sized) and len(items) == 0:
        return

    item_iterator: Iterator[T_Item] = iter(items)
    worker_limit = max(1, max_in_flight)
    worker_count = (
        min(len(items), worker_limit) if isinstance(items, Sized) else worker_limit
    )
    queue: asyncio.Queue[tuple[int, T_Item, T_Result | BaseException] | _WorkerDone] = (
        asyncio.Queue()
    )
    index_lock = asyncio.Lock()
    next_idx = 0

    async def worker() -> None:
        nonlocal next_idx
        try:
            while True:
                async with index_lock:
                    try:
                        item = next(item_iterator)
                    except StopIteration:
                        return
                    idx = next_idx
                    next_idx += 1

                try:
                    result = await process_one(idx, item, async_client)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    await queue.put((idx, item, exc))
                else:
                    await queue.put((idx, item, result))
        finally:
            await queue.put(_WorkerDone())

    tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]
    finished_workers = 0
    try:
        while finished_workers < worker_count:
            queue_item = await queue.get()
            if isinstance(queue_item, _WorkerDone):
                finished_workers += 1
                continue
            yield queue_item
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
