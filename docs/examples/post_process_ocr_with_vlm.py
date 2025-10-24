from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

from docling_core.types.doc import (
    DoclingDocument,
    ImageRefMode,
    NodeItem,
    PageItem,
    TextItem,
)
from docling_core.types.doc.document import (
    ContentLayer,
    DocItem,
    GraphCell,
    KeyValueItem,
    PictureItem,
    RichTableCell,
    TableCell,
    TableItem,
)
from PIL import Image
from PIL.ImageOps import crop
from pydantic import BaseModel, ConfigDict

from docling.backend.json.docling_json_backend import DoclingJSONBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat, ItemAndImageEnrichmentElement
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import BaseModelWithOptions, GenericEnrichmentModel
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.utils.api_image_request import api_image_request
from docling.utils.profiling import ProfilingScope, TimeRecorder
from docling.utils.utils import chunkify

# Example on how to apply to Docling Document OCR as a post-processing with "nanonets-ocr2-3b" via LM Studio
# Requires LM Studio running inference server with "nanonets-ocr2-3b" model pre-loaded
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "nanonets-ocr2-3b"
DEFAULT_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally."
)

PDF_DOC = "tests/data/pdf/2305.03393v1-pg9.pdf"
JSON_DOC = "scratch/test_doc.json"
POST_PROCESSED_JSON_DOC = "scratch/test_doc_ocr.json"


class OcrEnrichmentElement(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: Union[DocItem, TableCell, RichTableCell, GraphCell]
    image: list[
        Image.Image
    ]  # TODO maybe needs to be an array of images for multi-provenance things.


class OcrEnrichmentPipelineOptions(ConvertPipelineOptions):
    api_options: PictureDescriptionApiOptions


class OcrEnrichmentPipeline(SimplePipeline):
    def __init__(self, pipeline_options: OcrEnrichmentPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: OcrEnrichmentPipelineOptions

        self.enrichment_pipe = [
            OcrApiEnrichmentModel(
                enabled=True,
                enable_remote_services=True,
                artifacts_path=None,
                options=self.pipeline_options.api_options,
                accelerator_options=AcceleratorOptions(),
            )
        ]

    @classmethod
    def get_default_options(cls) -> OcrEnrichmentPipelineOptions:
        return OcrEnrichmentPipelineOptions()

    def _enrich_document(self, conv_res: ConversionResult) -> ConversionResult:
        def _prepare_elements(
            conv_res: ConversionResult, model: GenericEnrichmentModel[Any]
        ) -> Iterable[NodeItem]:
            for doc_element, _level in conv_res.document.iterate_items(
                traverse_pictures=True,
                included_content_layers={
                    ContentLayer.BODY,
                    ContentLayer.FURNITURE,
                },
            ):  # With all content layers, with traverse_pictures=True
                prepared_elements = (
                    model.prepare_element(  # make this one yield multiple items.
                        conv_res=conv_res, element=doc_element
                    )
                )
                if prepared_elements is not None:
                    yield prepared_elements

        with TimeRecorder(conv_res, "doc_enrich", scope=ProfilingScope.DOCUMENT):
            for model in self.enrichment_pipe:
                for element_batch in chunkify(
                    _prepare_elements(conv_res, model),
                    model.elements_batch_size,
                ):
                    for element in model(
                        doc=conv_res.document, element_batch=element_batch
                    ):  # Must exhaust!
                        pass
        return conv_res


class OcrApiEnrichmentModel(
    GenericEnrichmentModel[OcrEnrichmentElement], BaseModelWithOptions
):
    expansion_factor: float = 0.001

    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> Optional[list[OcrEnrichmentElement]]:
        if not self.is_processable(doc=conv_res.document, element=element):
            return None

        # allowed = (DocItem, TableCell, RichTableCell, GraphCell)
        allowed = (DocItem, TableItem, GraphCell)
        assert isinstance(
            element, allowed
        )  # too strict, could be DocItem, TableCell, RichTableCell, GraphCell

        if isinstance(element, KeyValueItem):
            # Yield from the graphCells inside here.
            result = []
            for c in element.graph.cells:
                element_prov = c.prov  # Key / Value have only one provenance!
                bbox = element_prov.bbox
                page_ix = element_prov.page_no
                bbox = bbox.scale_to_size(
                    old_size=conv_res.document.pages[page_ix].size,
                    new_size=conv_res.document.pages[page_ix].image.size,
                )
                expanded_bbox = bbox.expand_by_scale(
                    x_scale=self.expansion_factor, y_scale=self.expansion_factor
                ).to_top_left_origin(
                    page_height=conv_res.document.pages[page_ix].image.size.height
                )

                good_bbox = True
                if (
                    expanded_bbox.l > expanded_bbox.r
                    or expanded_bbox.t > expanded_bbox.b
                ):
                    good_bbox = False

                if good_bbox:
                    cropped_image = conv_res.document.pages[
                        page_ix
                    ].image.pil_image.crop(expanded_bbox.as_tuple())
                    # cropped_image.show()
                    result.append(OcrEnrichmentElement(item=c, image=[cropped_image]))
            return result
        elif isinstance(element, TableItem):
            element_prov = element.prov[0]
            page_ix = element_prov.page_no
            result = []
            for i, row in enumerate(element.data.grid):
                for j, cell in enumerate(row):
                    if hasattr(cell, "bbox"):
                        if cell.bbox:
                            bbox = cell.bbox
                            bbox = bbox.scale_to_size(
                                old_size=conv_res.document.pages[page_ix].size,
                                new_size=conv_res.document.pages[page_ix].image.size,
                            )
                            expanded_bbox = bbox.expand_by_scale(
                                x_scale=self.expansion_factor,
                                y_scale=self.expansion_factor,
                            ).to_top_left_origin(
                                page_height=conv_res.document.pages[
                                    page_ix
                                ].image.size.height
                            )

                            good_bbox = True
                            if (
                                expanded_bbox.l > expanded_bbox.r
                                or expanded_bbox.t > expanded_bbox.b
                            ):
                                good_bbox = False

                            if good_bbox:
                                cropped_image = conv_res.document.pages[
                                    page_ix
                                ].image.pil_image.crop(expanded_bbox.as_tuple())
                                # cropped_image.show()
                                result.append(
                                    OcrEnrichmentElement(
                                        item=cell, image=[cropped_image]
                                    )
                                )
            return result
        else:
            multiple_crops = []
            # Crop the image form the page
            for element_prov in element.prov:
                # element_prov = element.prov[0] # TODO: Not all items have prov
                bbox = element_prov.bbox

                page_ix = element_prov.page_no
                bbox = bbox.scale_to_size(
                    old_size=conv_res.document.pages[page_ix].size,
                    new_size=conv_res.document.pages[page_ix].image.size,
                )
                expanded_bbox = bbox.expand_by_scale(
                    x_scale=self.expansion_factor, y_scale=self.expansion_factor
                ).to_top_left_origin(
                    page_height=conv_res.document.pages[page_ix].image.size.height
                )

                good_bbox = True
                if (
                    expanded_bbox.l > expanded_bbox.r
                    or expanded_bbox.t > expanded_bbox.b
                ):
                    good_bbox = False

                if good_bbox:
                    cropped_image = conv_res.document.pages[
                        page_ix
                    ].image.pil_image.crop(expanded_bbox.as_tuple())
                    multiple_crops.append(cropped_image)
                    # cropped_image.show()
                    # Return the proper cropped image
                    multiple_crops
            if len(multiple_crops) > 0:
                return [OcrEnrichmentElement(item=element, image=multiple_crops)]
            else:
                return []

    @classmethod
    def get_options_type(cls) -> type[PictureDescriptionApiOptions]:
        return PictureDescriptionApiOptions

    def __init__(
        self,
        *,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionApiOptions,
        accelerator_options: AcceleratorOptions,
    ):
        self.enabled = enabled
        self.options = options
        self.concurrency = 4
        self.expansion_factor = 0.05
        self.elements_batch_size = 4
        self._accelerator_options = accelerator_options
        self._artifacts_path = (
            Path(artifacts_path) if isinstance(artifacts_path, str) else artifacts_path
        )

        if self.enabled and not enable_remote_services:
            raise OperationNotAllowed(
                "Enable remote services by setting pipeline_options.enable_remote_services=True."
            )

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        def _api_request(image: Image.Image) -> str:
            return api_image_request(
                image=image,
                prompt=self.options.prompt,
                url=self.options.url,
                timeout=self.options.timeout,
                headers=self.options.headers,
                **self.options.params,
            )

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            yield from executor.map(_api_request, images)

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        elements: list[TextItem] = []
        images: list[Image.Image] = []
        img_ind_per_element: list[int] = []

        for element_stack in element_batch:
            for element in element_stack:
                allowed = (DocItem, TableCell, RichTableCell, GraphCell)
                assert isinstance(element.item, allowed)
                for ind, img in enumerate(element.image):
                    elements.append(element.item)
                    images.append(img)
                    # images.append(element.image)
                    img_ind_per_element.append(ind)

        if not images:
            return

        outputs = list(self._annotate_images(images))

        for item, output, img_ind in zip(elements, outputs, img_ind_per_element):
            # Sometimes model can return html tags, which are not strictly needed in our, so it's better to clean them
            def clean_html_tags(text):
                for tag in [
                    "<table>",
                    "<tr>",
                    "<td>",
                    "<strong>",
                    "</table>",
                    "</tr>",
                    "</td>",
                    "</strong>",
                    "<th>",
                    "</th>",
                    "<tbody>",
                    "<tbody>",
                    "<thead>",
                    "</thead>",
                ]:
                    text = text.replace(tag, "")
                return text

            output = clean_html_tags(output)

            if isinstance(item, (TextItem)):
                print(f"OLD TEXT: {item.text}")
                print(f"NEW TEXT: {output}")

            # Re-populate text
            if isinstance(item, (TextItem, GraphCell)):
                if img_ind > 0:
                    # Concat texts across several provenances
                    item.text += " " + output
                    item.orig += " " + output
                else:
                    item.text = output
                    item.orig = output
            elif isinstance(item, (TableCell, RichTableCell)):
                item.text = output
            elif isinstance(item, PictureItem):
                pass
            else:
                raise ValueError(f"Unknown item type: {type(item)}")

            # Take care of charspans for relevant types
            if isinstance(item, GraphCell):
                item.prov.charspan = [0, len(output)]
            elif isinstance(item, TextItem):
                item.prov[0].charspan = [0, len(output)]

            yield item


def main() -> None:
    # TODO: Properly process cases for the items which have more than one provenance

    # Let's prepare a Docling document json with embedded page images
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 4.0

    doc_converter = (
        DocumentConverter(  # all of the below is optional, has internal defaults.
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline, pipeline_options=pipeline_options
                )
            },
        )
    )

    print("Converting PDF to get a Docling document json with embedded page images...")
    conv_result = doc_converter.convert(PDF_DOC)
    conv_result.document.save_as_json(
        filename=JSON_DOC, image_mode=ImageRefMode.EMBEDDED
    )

    md1 = conv_result.document.export_to_markdown()
    print("*** ORIGINAL MARKDOWN ***")
    print(md1)

    print("Post-process all bounding boxes with OCR")
    # Post-Process OCR on top of existing Docling document:
    pipeline_options = OcrEnrichmentPipelineOptions(
        api_options=PictureDescriptionApiOptions(
            url=LM_STUDIO_URL,
            prompt=DEFAULT_PROMPT,
            provenance="lm-studio-ocr",
            batch_size=4,
            concurrency=2,
            scale=2.0,
            params={"model": LM_STUDIO_MODEL},
        )
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.JSON_DOCLING: FormatOption(
                pipeline_cls=OcrEnrichmentPipeline,
                pipeline_options=pipeline_options,
                backend=DoclingJSONBackend,
            )
        }
    )
    result = doc_converter.convert(JSON_DOC)
    result.document.pages[1].image.pil_image.show()
    result.document.save_as_json(POST_PROCESSED_JSON_DOC)
    md = result.document.export_to_markdown()
    print("*** MARKDOWN ***")
    print(md)
    # print("*** KV ITEMS ***")
    # for kv_item in result.document.key_value_items:
    #     for kv_item_cell in kv_item.graph.cells:
    #         print("{} - {}".format(kv_item_cell.label, kv_item_cell.text))


if __name__ == "__main__":
    main()
