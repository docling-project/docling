from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, AsyncIterable, Callable, Coroutine, Dict, List

from docling.datamodel.document import ConversionResult, InputDocument, Page
from docling.pipeline.graph import STOP_SENTINEL, PipelineStage, StreamItem


class SourceStage(PipelineStage):
    """A placeholder stage to represent the entry point of the graph."""

    async def run(self) -> None:
        # This stage is driven by the GraphRunner's _run_source method
        # and does not have its own execution loop.
        pass


class SinkStage(PipelineStage):
    """A placeholder stage to represent the exit point of the graph."""

    async def run(self) -> None:
        # This stage is read by the GraphRunner's _run_sink method
        # and does not have its own execution loop.
        pass


class ExtractionStage(PipelineStage):
    """Extracts pages from documents and tracks them."""

    def __init__(
        self,
        name: str,
        page_tracker: Any,
        max_concurrent_extractions: int,
    ):
        super().__init__(name)
        self.page_tracker = page_tracker
        self.semaphore = asyncio.Semaphore(max_concurrent_extractions)
        self.input_channel = "in"
        self.output_channel = "out"
        self.failure_channel = "fail"

    async def _extract_page(
        self, page_no: int, conv_res: ConversionResult
    ) -> StreamItem | None:
        """Coroutine to extract a single page."""
        try:
            async with self.semaphore:
                page = Page(page_no=page_no)
                conv_res.pages.append(page)

                page._backend = await self.loop.run_in_executor(
                    None, conv_res.input._backend.load_page, page_no
                )

                if page._backend and page._backend.is_valid():
                    page.size = page._backend.get_size()
                    await self.page_tracker.track_page_loaded(page, conv_res)
                    return StreamItem(
                        payload=page, conv_res_id=id(conv_res), conv_res=conv_res
                    )
        except Exception:
            # In case of page-level error, we might log it but continue
            # For now, we don't propagate failure here, but in the document level
            pass
        return None

    async def _process_document(self, in_doc: InputDocument):
        """Processes a single document, extracting all its pages."""
        conv_res = ConversionResult(input=in_doc)

        try:
            from docling.backend.pdf_backend import PdfDocumentBackend

            if not isinstance(in_doc._backend, PdfDocumentBackend):
                raise TypeError("Backend is not a valid PdfDocumentBackend")

            total_pages = in_doc.page_count
            await self.page_tracker.register_document(conv_res, total_pages)

            start_page, end_page = conv_res.input.limits.page_range
            page_indices_to_extract = [
                i for i in range(total_pages) if (start_page - 1) <= i <= (end_page - 1)
            ]

            tasks = [
                self.loop.create_task(self._extract_page(i, conv_res))
                for i in page_indices_to_extract
            ]
            pages_extracted = await asyncio.gather(*tasks)

            await self._send_to_outputs(
                self.output_channel, [p for p in pages_extracted if p]
            )

        except Exception:
            conv_res.status = "FAILURE"
            await self._send_to_outputs(self.failure_channel, [conv_res])

    async def run(self) -> None:
        """Main loop to consume documents and launch extraction tasks."""
        q_in = self.input_queues[self.input_channel]
        while True:
            doc = await q_in.get()
            if doc is STOP_SENTINEL:
                await self._signal_downstream_completion()
                break
            await self._process_document(doc)


class PageProcessorStage(PipelineStage):
    """Applies a synchronous, 1-to-1 processing function to each page."""

    def __init__(self, name: str, model: Any):
        super().__init__(name)
        self.model = model
        self.input_channel = "in"
        self.output_channel = "out"

    async def run(self) -> None:
        q_in = self.input_queues[self.input_channel]
        while True:
            item = await q_in.get()
            if item is STOP_SENTINEL:
                await self._signal_downstream_completion()
                break

            # The model call is sync, run in thread to avoid blocking event loop
            processed_page = await self.loop.run_in_executor(
                None, lambda: next(iter(self.model(item.conv_res, [item.payload])))
            )
            item.payload = processed_page
            await self._send_to_outputs(self.output_channel, [item])


class BatchProcessorStage(PipelineStage):
    """Batches items and applies a synchronous model to the batch."""

    def __init__(
        self,
        name: str,
        model: Any,
        batch_size: int,
        batch_timeout: float,
    ):
        super().__init__(name)
        self.model = model
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.input_channel = "in"
        self.output_channel = "out"

    async def _collect_batch(self, q_in: asyncio.Queue) -> List[StreamItem] | None:
        """Collects a batch of items from the input queue with a timeout."""
        try:
            # Wait for the first item without a timeout
            first_item = await q_in.get()
            if first_item is STOP_SENTINEL:
                return None  # End of stream
        except asyncio.CancelledError:
            return None

        batch = [first_item]
        start_time = self.loop.time()

        while len(batch) < self.batch_size:
            timeout = self.batch_timeout - (self.loop.time() - start_time)
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(q_in.get(), timeout)
                if item is STOP_SENTINEL:
                    # Put sentinel back for other potential consumers or the main loop
                    await q_in.put(STOP_SENTINEL)
                    break
                batch.append(item)
            except asyncio.TimeoutError:
                break  # Batching timeout reached
        return batch

    async def run(self) -> None:
        q_in = self.input_queues[self.input_channel]
        while True:
            batch = await self._collect_batch(q_in)

            if not batch:  # This can be None or an empty list.
                await self._signal_downstream_completion()
                break

            # Group pages by their original ConversionResult
            grouped_by_doc = defaultdict(list)
            for item in batch:
                grouped_by_doc[item.conv_res_id].append(item)

            processed_items = []
            for conv_res_id, items in grouped_by_doc.items():
                conv_res = items[0].conv_res
                pages = [item.payload for item in items]

                # The model call is sync, run in thread
                processed_pages = await self.loop.run_in_executor(
                    None, lambda: list(self.model(conv_res, pages))
                )

                # Re-wrap the processed pages into StreamItems
                for i, page in enumerate(processed_pages):
                    processed_items.append(
                        StreamItem(
                            payload=page,
                            conv_res_id=items[i].conv_res_id,
                            conv_res=items[i].conv_res,
                        )
                    )

            await self._send_to_outputs(self.output_channel, processed_items)


class AggregationStage(PipelineStage):
    """Aggregates processed pages back into completed documents."""

    def __init__(
        self,
        name: str,
        page_tracker: Any,
        finalizer_func: Callable[[ConversionResult], Coroutine],
    ):
        super().__init__(name)
        self.page_tracker = page_tracker
        self.finalizer_func = finalizer_func
        self.success_channel = "in"
        self.failure_channel = "fail"
        self.output_channel = "out"

    async def run(self) -> None:
        success_q = self.input_queues[self.success_channel]
        failure_q = self.input_queues.get(self.failure_channel)
        doc_completers: Dict[int, asyncio.Future] = {}

        async def handle_successes():
            while True:
                item = await success_q.get()
                if item is STOP_SENTINEL:
                    break
                is_doc_complete = await self.page_tracker.track_page_completion(
                    item.payload, item.conv_res
                )
                if is_doc_complete:
                    await self.finalizer_func(item.conv_res)
                    await self._send_to_outputs(self.output_channel, [item.conv_res])
                    if item.conv_res_id in doc_completers:
                        doc_completers.pop(item.conv_res_id).set_result(True)

        async def handle_failures():
            if failure_q is None:
                return  # No failure channel, nothing to do
            while True:
                failed_res = await failure_q.get()
                if failed_res is STOP_SENTINEL:
                    break
                await self._send_to_outputs(self.output_channel, [failed_res])
                if id(failed_res) in doc_completers:
                    doc_completers.pop(id(failed_res)).set_result(True)

        # Create tasks only for channels that exist
        tasks = [handle_successes()]
        if failure_q is not None:
            tasks.append(handle_failures())

        await asyncio.gather(*tasks)
        await self._signal_downstream_completion()
