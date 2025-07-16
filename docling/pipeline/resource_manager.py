import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.pipeline.async_base_pipeline import DocumentTracker

_log = logging.getLogger(__name__)


@dataclass
class AsyncPageTracker:
    """Manages page backend lifecycle across documents"""

    _doc_trackers: Dict[str, DocumentTracker] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    keep_images: bool = False
    keep_backend: bool = False

    async def register_document(
        self, conv_res: ConversionResult, total_pages: int
    ) -> str:
        """Register a new document for tracking"""
        async with self._lock:
            doc_id = str(id(conv_res))
            self._doc_trackers[doc_id] = DocumentTracker(
                doc_id=doc_id, total_pages=total_pages, conv_result=conv_res
            )
            return doc_id

    async def track_page_loaded(self, page: Page, conv_res: ConversionResult) -> None:
        """Track when a page backend is loaded"""
        async with self._lock:
            doc_id = str(id(conv_res))
            if doc_id in self._doc_trackers and page._backend is not None:
                self._doc_trackers[doc_id].page_backends[page.page_no] = page._backend

    async def track_page_completion(
        self, page: Page, conv_res: ConversionResult
    ) -> bool:
        """Track page completion and cleanup when all pages done"""
        async with self._lock:
            doc_id = str(id(conv_res))
            if doc_id not in self._doc_trackers:
                _log.warning(f"Document {doc_id} not registered for tracking")
                return False

            tracker = self._doc_trackers[doc_id]
            tracker.processed_pages += 1

            # Clear this page's image cache if needed
            if not self.keep_images:
                page._image_cache = {}

            # If all pages from this document are processed, cleanup
            if tracker.processed_pages == tracker.total_pages:
                await self._cleanup_document_resources(tracker)
                del self._doc_trackers[doc_id]
                return True  # Document is complete

            return False  # Document is not yet complete

    async def _cleanup_document_resources(self, tracker: DocumentTracker) -> None:
        """Cleanup all resources for a completed document"""
        if not self.keep_backend:
            # Unload all page backends for this document
            for page_no, backend in tracker.page_backends.items():
                if backend is not None:
                    try:
                        # Run unload in thread to avoid blocking
                        await asyncio.to_thread(backend.unload)
                    except Exception as e:
                        _log.warning(
                            f"Failed to unload backend for page {page_no}: {e}"
                        )

        tracker.page_backends.clear()
        _log.debug(f"Cleaned up resources for document {tracker.doc_id}")

    async def cleanup_all(self) -> None:
        """Cleanup all tracked documents - for shutdown"""
        async with self._lock:
            for tracker in self._doc_trackers.values():
                await self._cleanup_document_resources(tracker)
            self._doc_trackers.clear()


@dataclass
class ConversionResultAccumulator:
    """Accumulates updates to ConversionResult without immediate mutation"""

    _updates: Dict[str, Dict] = field(default_factory=dict)  # doc_id -> updates
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def accumulate_page_result(
        self, page_no: int, conv_res: ConversionResult, updates: Dict
    ) -> None:
        """Accumulate updates for later application"""
        async with self._lock:
            doc_id = str(id(conv_res))
            if doc_id not in self._updates:
                self._updates[doc_id] = {}

            if page_no not in self._updates[doc_id]:
                self._updates[doc_id][page_no] = {}

            self._updates[doc_id][page_no].update(updates)

    async def flush_to_conv_res(self, conv_res: ConversionResult) -> None:
        """Apply all accumulated updates atomically"""
        async with self._lock:
            doc_id = str(id(conv_res))
            if doc_id in self._updates:
                # Apply updates
                for page_no, updates in self._updates[doc_id].items():
                    # Find the page and apply updates
                    for page in conv_res.pages:
                        if page.page_no == page_no:
                            for key, value in updates.items():
                                setattr(page, key, value)
                            break

                del self._updates[doc_id]
