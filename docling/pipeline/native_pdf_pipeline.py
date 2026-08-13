import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    ProvenanceItem,
    Size,
)
from docling_core.types.doc.page import SegmentedPdfPage

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    Page,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import NativePdfPipelineOptions
from docling.pipeline.base_pipeline import (
    ConvertPipeline,
    get_expected_page_nos,
    iter_requested_page_backends,
)
from docling.utils.profiling import (
    ProfilingScope,
    TimeIntervalRecorder,
    TimeRecorder,
)

_log = logging.getLogger(__name__)


@dataclass
class _NativeTimings:
    """Where a single document spent its time, in seconds.

    Recorded unconditionally (the clock reads cost nothing next to the work they
    measure) so a slow conversion can be explained from the logs alone, without
    re-running it under `--profiling`.

    The parser decodes and rasterizes a page in one shot on its worker threads,
    so ``parser_wait`` - blocking until the next page comes out - is the only
    phase that shrinks when more parser threads are given. Every other phase runs
    on the calling thread and is unaffected by the thread count.
    """

    parser_wait: float = 0.0
    materialize: float = 0.0
    page_image_fetch: float = 0.0
    page_image_encode: float = 0.0
    text_items: float = 0.0
    picture_items: float = 0.0

    n_text_items: int = 0
    n_pictures: int = 0
    n_picture_images: int = 0

    def record(self, conv_res: ConversionResult) -> None:
        """Publish the phases as timings, so `--profiling` reports them too."""
        for key, seconds in (
            ("native_parser_wait", self.parser_wait),
            ("native_materialize", self.materialize),
            ("native_page_image_fetch", self.page_image_fetch),
            ("native_page_image_encode", self.page_image_encode),
            ("native_text_items", self.text_items),
            ("native_picture_items", self.picture_items),
        ):
            recorder = TimeIntervalRecorder(
                conv_res, key, scope=ProfilingScope.DOCUMENT
            )
            recorder.add(seconds)
            recorder.close()


class NativePdfPipeline(ConvertPipeline):
    """Model-free PDF pipeline built on the native content of the PDF.

    Every text cell reported by the PDF backend becomes a plain `TextItem` with a
    provenance box, every embedded bitmap becomes a `PictureItem`, and - when
    `generate_page_images` is set - the rendered page is attached to the page. No
    layout, OCR or table model runs, so conversion is fast, but the document has
    no reading order, headings or tables: items appear in the order the parser
    reports them.
    """

    def __init__(self, pipeline_options: NativePdfPipelineOptions) -> None:
        super().__init__(pipeline_options)
        self.pipeline_options: NativePdfPipelineOptions = pipeline_options
        self.keep_images = pipeline_options.generate_page_images
        self._timings = _NativeTimings()

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        backend = conv_res.input._backend
        if not isinstance(backend, PdfDocumentBackend):
            raise RuntimeError(
                f"The selected backend {type(backend).__name__} for {conv_res.input.file} "
                f"is not a PDF backend. Can not convert this with the native PDF pipeline. "
                f"Please check your format configuration on DocumentConverter."
            )

        expected_page_nos = get_expected_page_nos(conv_res)
        if not expected_page_nos:
            conv_res.status = ConversionStatus.FAILURE
            return conv_res

        self._timings = _NativeTimings()
        self._warn_on_rerender(backend)

        start_time = time.monotonic()
        with TimeRecorder(conv_res, "doc_build", scope=ProfilingScope.DOCUMENT):
            for page_backend in self._timed_page_backends(
                iter_requested_page_backends(backend, expected_page_nos)
            ):
                page = self._parse_page(conv_res, page_backend)
                if page is not None:
                    conv_res.pages.append(page)
                page_backend.unload()

                timeout = self.pipeline_options.document_timeout
                elapsed = time.monotonic() - start_time
                if timeout is not None and elapsed > timeout:
                    timeout_msg = (
                        f"Document processing timeout: exceeded {timeout:.3f}s limit after "
                        f"{elapsed:.3f}s. Processed {len(conv_res.pages)}/"
                        f"{len(expected_page_nos)} pages."
                    )
                    _log.warning(timeout_msg)
                    conv_res.errors.append(
                        ErrorItem(
                            component_type=DoclingComponentType.PIPELINE,
                            module_name=self.__class__.__name__,
                            error_message=timeout_msg,
                            category=FailureCategory.TIMEOUT,
                        )
                    )
                    conv_res.status = ConversionStatus.PARTIAL_SUCCESS
                    break

            # Backends without random page access (the threaded docling-parse
            # backend) deliver their pages in completion order, so the parsed
            # pages are collected first and only then put back in document order
            # for the assembly step.
            conv_res.pages.sort(key=lambda page: page.page_no)

        _log.info(
            "Native build of %s: %d page(s) in %.2fs with %d parser thread(s) "
            "[waiting for the parser %.2fs, materializing cells %.2fs, "
            "page images %.2fs]",
            conv_res.input.file.name,
            len(conv_res.pages),
            time.monotonic() - start_time,
            self.pipeline_options.parser_threads,
            self._timings.parser_wait,
            self._timings.materialize,
            self._timings.page_image_fetch,
        )

        if not conv_res.pages:
            conv_res.status = ConversionStatus.FAILURE

        return conv_res

    def _timed_page_backends(
        self, page_backends: Iterator[PdfPageBackend]
    ) -> Iterator[PdfPageBackend]:
        """Yield the page backends, timing how long each one is waited for.

        That wait is the pipeline's view of the parser's threaded decode: with
        enough parser threads the next page is already finished and the wait is
        near zero, while a single thread makes it the bulk of the build.
        """
        while True:
            started = time.monotonic()
            try:
                page_backend = next(page_backends)
            except StopIteration:
                return
            finally:
                self._timings.parser_wait += time.monotonic() - started
            yield page_backend

    def _warn_on_rerender(self, backend: PdfDocumentBackend) -> None:
        """Warn when every page image will be rasterized a second time.

        The threaded parser rasterizes each page on its worker threads while
        decoding it, but only at its own `render_scale`. Asking for a page image
        at any other scale re-rasterizes it here, on the calling thread, which is
        both duplicate work and serial.
        """
        if not self.pipeline_options.generate_page_images:
            return
        options = backend.options
        if not isinstance(options, ThreadedDoclingParseBackendOptions):
            return
        if not options.render_pages:
            return
        if math.isclose(options.render_scale, self.pipeline_options.images_scale):
            return

        _log.warning(
            "The backend rasterizes pages at scale %.2f but the pipeline asks for "
            "page images at scale %.2f, so every page is rasterized twice, the "
            "second time serially. Set the backend's render_scale to %.2f.",
            options.render_scale,
            self.pipeline_options.images_scale,
            self.pipeline_options.images_scale,
        )

    def _parse_page(
        self, conv_res: ConversionResult, page_backend: PdfPageBackend
    ) -> Page | None:
        """Collect the native content of one page, or None if it failed to parse."""
        if not page_backend.is_valid():
            detail = page_backend.get_error_message()
            self._record_page_failure(
                conv_res,
                page_backend,
                f"Page failed to parse: {detail}"
                if detail
                else "Page failed to parse.",
            )
            return None

        try:
            started = time.monotonic()
            page = Page(page_no=page_backend.page_no)
            page.size = page_backend.get_size()
            page.parsed_page = page_backend.get_segmented_page()
            self._timings.materialize += time.monotonic() - started

            if self.pipeline_options.generate_page_images:
                # Cache the page image now: the page backend is released as soon as
                # the page is parsed, so it can not be rasterized later on.
                started = time.monotonic()
                page._backend = page_backend
                page._default_image_scale = self.pipeline_options.images_scale
                page.get_image(scale=self.pipeline_options.images_scale)
                page._backend = None
                self._timings.page_image_fetch += time.monotonic() - started
        except Exception as exc:
            # One unreadable page should not cost the whole document.
            self._record_page_failure(
                conv_res, page_backend, f"Page failed to parse: {exc}"
            )
            return None

        return page

    def _record_page_failure(
        self, conv_res: ConversionResult, page_backend: PdfPageBackend, message: str
    ) -> None:
        _log.warning(
            "Page %d of %s: %s", page_backend.page_no, conv_res.input.file, message
        )
        conv_res.errors.append(
            ErrorItem(
                component_type=DoclingComponentType.DOCUMENT_BACKEND,
                module_name=type(page_backend).__name__,
                error_message=message,
                category=FailureCategory.BACKEND_FAILURE,
                page_no=page_backend.page_no,
            )
        )
        conv_res.status = ConversionStatus.PARTIAL_SUCCESS

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            origin = DocumentOrigin(
                mimetype="application/pdf",
                filename=conv_res.input.file.name,
                binary_hash=conv_res.input.document_hash,
            )
            doc = DoclingDocument(name=Path(origin.filename).stem, origin=origin)
            dpi = round(72 * self.pipeline_options.images_scale)

            for page in conv_res.pages:
                assert page.size is not None, "Page size is not initialized."

                started = time.monotonic()
                image = page.get_image(scale=self.pipeline_options.images_scale)
                doc.add_page(
                    page_no=page.page_no,
                    size=page.size,
                    image=ImageRef.from_pil(image, dpi=dpi)
                    if image is not None
                    else None,
                )
                self._timings.page_image_encode += time.monotonic() - started

                if page.parsed_page is None:
                    continue

                started = time.monotonic()
                self._add_text_items(doc, page.page_no, page.parsed_page, page.size)
                self._timings.text_items += time.monotonic() - started

                started = time.monotonic()
                self._add_picture_items(doc, page.page_no, page.parsed_page, page.size)
                self._timings.picture_items += time.monotonic() - started

            conv_res.document = doc

        timings = self._timings
        _log.info(
            "Native assembly of %s: %d text item(s), %d picture(s) (%d with an image) "
            "[page images %.2fs, text %.2fs, pictures %.2fs]",
            conv_res.input.file.name,
            timings.n_text_items,
            timings.n_pictures,
            timings.n_picture_images,
            timings.page_image_encode,
            timings.text_items,
            timings.picture_items,
        )
        timings.record(conv_res)

        if not self.keep_images:
            for page in conv_res.pages:
                page._image_cache = {}

        return conv_res

    def _add_text_items(
        self,
        doc: DoclingDocument,
        page_no: int,
        parsed_page: SegmentedPdfPage,
        size: Size,
    ) -> None:
        unit = self.pipeline_options.text_cell_unit
        n_cells = 0
        for cell in parsed_page.iterate_cells(unit):
            text = cell.text
            if not text:
                continue
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=text,
                orig=text,
                prov=ProvenanceItem(
                    page_no=page_no,
                    bbox=self._to_prov_bbox(cell.rect.to_bounding_box(), size),
                    charspan=(0, len(text)),
                ),
            )
            n_cells += 1

        self._timings.n_text_items += n_cells
        if n_cells == 0 and parsed_page.textline_cells:
            _log.warning(
                "Page %d has text, but no %s cells: the PDF backend does not materialize "
                "that text cell unit.",
                page_no,
                unit.value,
            )

    def _add_picture_items(
        self,
        doc: DoclingDocument,
        page_no: int,
        parsed_page: SegmentedPdfPage,
        size: Size,
    ) -> None:
        for bitmap in parsed_page.bitmap_resources:
            image = (
                bitmap.image if self.pipeline_options.generate_picture_images else None
            )
            doc.add_picture(
                image=image,
                prov=ProvenanceItem(
                    page_no=page_no,
                    bbox=self._to_prov_bbox(bitmap.rect.to_bounding_box(), size),
                    charspan=(0, 0),
                ),
            )
            self._timings.n_pictures += 1
            self._timings.n_picture_images += image is not None

    @staticmethod
    def _to_prov_bbox(bbox: BoundingBox, size: Size) -> BoundingBox:
        """Provenance boxes are bottom-left; parsed cells can be either origin."""
        if bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
            return bbox
        return bbox.to_bottom_left_origin(page_height=size.height)

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        if conv_res.status in (ConversionStatus.PENDING, ConversionStatus.STARTED):
            return ConversionStatus.SUCCESS
        return conv_res.status

    def _unload(self, conv_res: ConversionResult) -> None:
        for page in conv_res.pages:
            if page._backend is not None:
                page._backend.unload()
        if conv_res.input._backend:
            conv_res.input._backend.unload()

    @classmethod
    def get_default_options(cls) -> NativePdfPipelineOptions:
        return NativePdfPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend) -> bool:
        return isinstance(backend, PdfDocumentBackend)
