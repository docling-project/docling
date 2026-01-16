import logging
import warnings
from pathlib import Path
from typing import Optional, cast

import numpy as np
from docling_core.types.doc import DocItem, ImageRef, PictureItem, TableItem

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import AssembledUnit, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.layout_model_specs import LayoutModelConfig
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.factories import (
    get_layout_factory,
    get_ocr_factory,
    get_table_structure_factory,
)
from docling.models.stages.code_formula.code_formula_model import (
    CodeFormulaModel,
    CodeFormulaModelOptions,
)
from docling.models.stages.page_assemble.page_assemble_model import (
    PageAssembleModel,
    PageAssembleOptions,
)
from docling.models.stages.page_preprocessing.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)
from docling.pipeline.base_pipeline import PaginatedPipeline
from docling.utils.model_downloader import download_models
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class LegacyStandardPdfPipeline(PaginatedPipeline):
    def __init__(self, pipeline_options: PdfPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: PdfPipelineOptions

        with warnings.catch_warnings():  # deprecated generate_table_images
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.keep_images = (
                self.pipeline_options.generate_page_images
                or self.pipeline_options.generate_picture_images
                or self.pipeline_options.generate_table_images
            )

        self.reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())

        ocr_model = self.get_ocr_model(artifacts_path=self.artifacts_path)

        layout_factory = get_layout_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        layout_model = layout_factory.create_instance(
            options=pipeline_options.layout_options,
            artifacts_path=self.artifacts_path,
            accelerator_options=pipeline_options.accelerator_options,
        )
        table_factory = get_table_structure_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        table_model = table_factory.create_instance(
            options=pipeline_options.table_structure_options,
            enabled=pipeline_options.do_table_structure,
            artifacts_path=self.artifacts_path,
            accelerator_options=pipeline_options.accelerator_options,
        )

        self.build_pipe = [
            # Pre-processing
            PagePreprocessingModel(
                options=PagePreprocessingOptions(
                    images_scale=pipeline_options.images_scale,
                )
            ),
            # OCR
            ocr_model,
            # Layout model
            layout_model,
            # Table structure model
            table_model,
            # Page assemble
            PageAssembleModel(options=PageAssembleOptions()),
        ]

        self.enrichment_pipe = [
            # Code Formula Enrichment Model
            CodeFormulaModel(
                enabled=pipeline_options.do_code_enrichment
                or pipeline_options.do_formula_enrichment,
                artifacts_path=self.artifacts_path,
                options=CodeFormulaModelOptions(
                    do_code_enrichment=pipeline_options.do_code_enrichment,
                    do_formula_enrichment=pipeline_options.do_formula_enrichment,
                ),
                accelerator_options=pipeline_options.accelerator_options,
            ),
            *self.enrichment_pipe,
        ]

        if (
            self.pipeline_options.do_formula_enrichment
            or self.pipeline_options.do_code_enrichment
            or self.pipeline_options.do_picture_classification
            or self.pipeline_options.do_picture_description
        ):
            self.keep_backend = True

    @staticmethod
    def download_models_hf(
        local_dir: Optional[Path] = None, force: bool = False
    ) -> Path:
        warnings.warn(
            "The usage of LegacyStandardPdfPipeline.download_models_hf() is deprecated "
            "use instead the utility `docling-tools models download`, or "
            "the upstream method docling.utils.models_downloader.download_all()",
            DeprecationWarning,
            stacklevel=3,
        )

        output_dir = download_models(output_dir=local_dir, force=force, progress=False)
        return output_dir

    def get_ocr_model(self, artifacts_path: Optional[Path] = None) -> BaseOcrModel:
        factory = get_ocr_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        return factory.create_instance(
            options=self.pipeline_options.ocr_options,
            enabled=self.pipeline_options.do_ocr,
            artifacts_path=artifacts_path,
            accelerator_options=self.pipeline_options.accelerator_options,
        )

    def initialize_page(self, conv_res: ConversionResult, page: Page) -> Page:
        with TimeRecorder(conv_res, "page_init"):
            page._backend = conv_res.input._backend.load_page(page.page_no - 1)  # type: ignore
            if page._backend is not None and page._backend.is_valid():
                page.size = page._backend.get_size()

        return page

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        all_elements = []
        all_headers = []
        all_body = []

        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            for p in conv_res.pages:
                if p.assembled is not None:
                    for el in p.assembled.body:
                        all_body.append(el)
                    for el in p.assembled.headers:
                        all_headers.append(el)
                    for el in p.assembled.elements:
                        all_elements.append(el)

            conv_res.assembled = AssembledUnit(
                elements=all_elements, headers=all_headers, body=all_body
            )

            conv_res.document = self.reading_order_model(conv_res)

            # Generate page images in the output
            if self.pipeline_options.generate_page_images:
                for page in conv_res.pages:
                    assert page.image is not None
                    page_no = page.page_no
                    conv_res.document.pages[page_no].image = ImageRef.from_pil(
                        page.image, dpi=int(72 * self.pipeline_options.images_scale)
                    )

            # Generate images of the requested element types
            with warnings.catch_warnings():  # deprecated generate_table_images
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                if (
                    self.pipeline_options.generate_picture_images
                    or self.pipeline_options.generate_table_images
                ):
                    scale = self.pipeline_options.images_scale
                    for element, _level in conv_res.document.iterate_items():
                        if not isinstance(element, DocItem) or len(element.prov) == 0:
                            continue
                        if (
                            isinstance(element, PictureItem)
                            and self.pipeline_options.generate_picture_images
                        ) or (
                            isinstance(element, TableItem)
                            and self.pipeline_options.generate_table_images
                        ):
                            page_ix = element.prov[0].page_no - 1
                            page = next(
                                (p for p in conv_res.pages if p.page_no == page_ix),
                                cast("Page", None),
                            )
                            assert page is not None
                            assert page.size is not None
                            assert page.image is not None

                            crop_bbox = (
                                element.prov[0]
                                .bbox.scaled(scale=scale)
                                .to_top_left_origin(
                                    page_height=page.size.height * scale
                                )
                            )

                            cropped_im = page.image.crop(crop_bbox.as_tuple())
                            element.image = ImageRef.from_pil(
                                cropped_im, dpi=int(72 * scale)
                            )

            # Aggregate confidence values for document:
            if len(conv_res.pages) > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning,
                        message="Mean of empty slice|All-NaN slice encountered",
                    )
                    conv_res.confidence.layout_score = float(
                        np.nanmean(
                            [c.layout_score for c in conv_res.confidence.pages.values()]
                        )
                    )
                    conv_res.confidence.parse_score = float(
                        np.nanquantile(
                            [c.parse_score for c in conv_res.confidence.pages.values()],
                            q=0.1,  # parse score should relate to worst 10% of pages.
                        )
                    )
                    conv_res.confidence.table_score = float(
                        np.nanmean(
                            [c.table_score for c in conv_res.confidence.pages.values()]
                        )
                    )
                    conv_res.confidence.ocr_score = float(
                        np.nanmean(
                            [c.ocr_score for c in conv_res.confidence.pages.values()]
                        )
                    )

        return conv_res

    @classmethod
    def get_default_options(cls) -> PdfPipelineOptions:
        return PdfPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, PdfDocumentBackend)
