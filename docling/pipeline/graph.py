from __future__ import annotations

import asyncio
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, AsyncIterable, Dict, List, Literal, Optional

# Sentinel to signal stream completion
STOP_SENTINEL = object()

# Global thread pool for pipeline operations - shared across all stages
_PIPELINE_THREAD_POOL: Optional[ThreadPoolExecutor] = None
_THREAD_POOL_REFS = weakref.WeakSet()


def get_pipeline_thread_pool(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    """Get or create the shared pipeline thread pool."""
    global _PIPELINE_THREAD_POOL
    if _PIPELINE_THREAD_POOL is None or _PIPELINE_THREAD_POOL._shutdown:
        _PIPELINE_THREAD_POOL = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="docling_pipeline"
        )
    _THREAD_POOL_REFS.add(_PIPELINE_THREAD_POOL)
    return _PIPELINE_THREAD_POOL


def shutdown_pipeline_thread_pool(wait: bool = True) -> None:
    """Shutdown the shared thread pool."""
    global _PIPELINE_THREAD_POOL
    if _PIPELINE_THREAD_POOL is not None:
        _PIPELINE_THREAD_POOL.shutdown(wait=wait)
        _PIPELINE_THREAD_POOL = None


@dataclass(slots=True)
class StreamItem:
    """
    A wrapper for data flowing through the pipeline, maintaining a link
    to the original conversion result context.
    """

    payload: Any
    conv_res_id: int
    conv_res: Any  # Opaque reference to ConversionResult


class PipelineStage(ABC):
    """A single, encapsulated step in a processing pipeline graph."""

    def __init__(self, name: str, max_workers: Optional[int] = None):
        self.name = name
        self.input_queues: Dict[str, asyncio.Queue] = {}
        self.output_queues: Dict[str, List[asyncio.Queue]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread_pool = get_pipeline_thread_pool(max_workers)

    @abstractmethod
    async def run(self) -> None:
        """
        The core execution logic for the stage. This method is responsible for
        consuming from input queues, processing data, and putting results into
        output queues.
        """

    async def _send_to_outputs(self, channel: str, items: List[StreamItem] | List[Any]):
        """Helper to send processed items to all connected output queues."""
        if channel in self.output_queues:
            for queue in self.output_queues[channel]:
                for item in items:
                    await queue.put(item)

    async def _signal_downstream_completion(self):
        """Signal that this stage is done processing to all output channels."""
        for channel_queues in self.output_queues.values():
            for queue in channel_queues:
                await queue.put(STOP_SENTINEL)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return self._loop

    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        """Get the shared thread pool for this stage."""
        return self._thread_pool


class GraphRunner:
    """Connects stages and runs the pipeline graph."""

    def __init__(self, stages: List[PipelineStage], edges: List[Dict[str, str]]):
        self._stages = {s.name: s for s in stages}
        self._edges = edges

    def _wire_graph(self, queue_max_size: int):
        """Create queues for edges and connect them to stage inputs and outputs."""
        for edge in self._edges:
            from_stage, from_output = edge["from_stage"], edge["from_output"]
            to_stage, to_input = edge["to_stage"], edge["to_input"]

            queue = asyncio.Queue(maxsize=queue_max_size)

            # Connect to source stage's output
            self._stages[from_stage].output_queues.setdefault(from_output, []).append(
                queue
            )

            # Connect to destination stage's input
            self._stages[to_stage].input_queues[to_input] = queue

    async def _run_source(
        self,
        source_stream: AsyncIterable[Any],
        source_stage: str,
        source_channel: str,
    ):
        """Feed the graph from an external async iterable."""
        output_queues = self._stages[source_stage].output_queues.get(source_channel, [])
        async for item in source_stream:
            for queue in output_queues:
                await queue.put(item)
        # Signal completion to all downstream queues
        for queue in output_queues:
            await queue.put(STOP_SENTINEL)

    async def _run_sink(self, sink_stage: str, sink_channel: str) -> AsyncIterable[Any]:
        """Yield results from the graph's final output queue."""
        queue = self._stages[sink_stage].input_queues[sink_channel]
        while True:
            item = await queue.get()
            if item is STOP_SENTINEL:
                break
            yield item
        await queue.put(STOP_SENTINEL)  # Allow other sinks to terminate

    async def run(
        self,
        source_stream: AsyncIterable,
        source_config: Dict[str, str],
        sink_config: Dict[str, str],
        queue_max_size: int = 32,
    ) -> AsyncIterable:
        """
        Executes the entire pipeline graph.

        Args:
            source_stream: The initial async iterable to feed the graph.
            source_config: Dictionary with "stage" and "channel" for the entry point.
            sink_config: Dictionary with "stage" and "channel" for the exit point.
            queue_max_size: The max size for the internal asyncio.Queues.
        """
        self._wire_graph(queue_max_size)

        try:
            async with asyncio.TaskGroup() as tg:
                # Create a task for the source feeder
                tg.create_task(
                    self._run_source(
                        source_stream, source_config["stage"], source_config["channel"]
                    )
                )

                # Create tasks for all pipeline stages
                for stage in self._stages.values():
                    tg.create_task(stage.run())

                # Yield results from the sink
                async for result in self._run_sink(
                    sink_config["stage"], sink_config["channel"]
                ):
                    yield result
        finally:
            # Ensure thread pool cleanup on pipeline completion
            # Note: We don't shutdown here as other pipelines might be using it
            pass
