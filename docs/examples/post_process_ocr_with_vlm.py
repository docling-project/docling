import argparse
import logging
import os
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

from docling_core.types.doc import (
    DoclingDocument,
    ImageRefMode,
    NodeItem,
    TextItem,
)
from docling_core.types.doc.document import (
    ContentLayer,
    DocItem,
    GraphCell,
    KeyValueItem,
    FormItem,
    PictureItem,
    RichTableCell,
    TableCell,
    TableItem,
)
from PIL import Image, ImageFilter
from PIL.ImageOps import crop
import numpy as np
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

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
# To run:
# uv run python docs/examples/post_process_ocr_with_vlm.py

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "nanonets-ocr2-3b"

DEFAULT_PROMPT = "Extract the text from the above document as if you were reading it naturally. Output pure text, no html and no markdown. Pay attention on line breaks and don't miss text after line break. Put all text in one line."
VERBOSE = True
SHOW_IMAGE = False
SHOW_EMPTY_CROPS = False
SHOW_NONEMPTY_CROPS = False
PRINT_RESULT_MARKDOWN = False

def is_empty_fast_with_lines_pil(
    pil_img: Image.Image,
    # downscale_max_side: int = 64,
    downscale_max_side: int = 48,
    grad_threshold: float = 15.0,   # how strong a gradient must be to count as edge
    min_line_coverage: float = 0.6, # line must cover 60% of height/width
    # max_allowed_lines: int = 4,     # allow up to this many strong lines
    max_allowed_lines: int = 10,     # allow up to this many strong lines
    edge_fraction_threshold: float = 0.0035
):
    """
    Fast 'empty' detector using only PIL + NumPy.

    Treats an image as empty if:
      - It has very few edges overall, OR
      - Edges can be explained by at most `max_allowed_lines` long vertical/horizontal lines.

    Returns:
      (is_empty: bool, remaining_edge_fraction: float, debug: dict)
    """

    # 1) Convert to grayscale
    gray = pil_img.convert("L")

    # 2) Aggressive downscale, keeping aspect ratio
    w0, h0 = gray.size
    max_side = max(w0, h0)
    if max_side > downscale_max_side:
        # scale = downscale_max_side / max_side
        # new_w = max(1, int(w0 * scale))
        # new_h = max(1, int(h0 * scale))

        new_w = downscale_max_side
        new_h = downscale_max_side

        gray = gray.resize((new_w, new_h), resample=Image.BILINEAR)

    w, h = gray.size
    if w == 0 or h == 0:
        return True, 0.0, {"reason": "zero_size"}

    # 3) Small blur to reduce noise
    gray = gray.filter(ImageFilter.BoxBlur(1))

    # 4) Convert to NumPy
    arr = np.asarray(gray, dtype=np.float32)  # shape (h, w) in PIL, but note: PIL size is (w, h)
    H, W = arr.shape
    total_pixels = H * W

    # 5) Compute simple gradients (forward differences)
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)

    gx[:, :-1] = arr[:, 1:] - arr[:, :-1]   # horizontal differences
    gy[:-1, :] = arr[1:, :] - arr[:-1, :]   # vertical differences

    mag = np.hypot(gx, gy)  # gradient magnitude

    # 6) Threshold gradients to get edges (boolean mask)
    edges = mag > grad_threshold
    edge_fraction = edges.mean()

    # Quick early-exit: almost no edges => empty
    if edge_fraction < edge_fraction_threshold:
        return True, float(edge_fraction), {"reason": "few_edges"}

    # 7) Detect strong vertical & horizontal lines via edge sums
    col_sum = edges.sum(axis=0)  # per column
    row_sum = edges.sum(axis=1)  # per row

    # Line must have edge pixels in at least `min_line_coverage` of the dimension
    vert_line_cols = np.where(col_sum >= min_line_coverage * H)[0]
    horiz_line_rows = np.where(row_sum >= min_line_coverage * W)[0]

    num_lines = len(vert_line_cols) + len(horiz_line_rows)

    # If we have more long lines than allowed => non-empty
    if num_lines > max_allowed_lines:
        return False, float(edge_fraction), {
            "reason": "too_many_lines",
            "num_lines": int(num_lines),
            "edge_fraction": float(edge_fraction),
        }

    # 8) Mask out those lines and recompute remaining edges
    line_mask = np.zeros_like(edges, dtype=bool)
    if len(vert_line_cols) > 0:
        line_mask[:, vert_line_cols] = True
    if len(horiz_line_rows) > 0:
        line_mask[horiz_line_rows, :] = True

    remaining_edges = edges & ~line_mask
    remaining_edge_fraction = remaining_edges.mean()

    is_empty = remaining_edge_fraction < edge_fraction_threshold

    debug = {
        "original_edge_fraction": float(edge_fraction),
        "remaining_edge_fraction": float(remaining_edge_fraction),
        "num_vert_lines": int(len(vert_line_cols)),
        "num_horiz_lines": int(len(horiz_line_rows)),
    }
    return is_empty, float(remaining_edge_fraction), debug


def remove_break_lines(text: str) -> str:
    # Replace any newline types with a single space
    cleaned = re.sub(r'[\r\n]+', ' ', text)
    # Collapse multiple spaces into one
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


def safe_crop(img: Image.Image, bbox):
    left, top, right, bottom = bbox
    # Clamp to image boundaries
    left   = max(0, min(left,   img.width))
    top    = max(0, min(top,    img.height))
    right  = max(0, min(right,  img.width))
    bottom = max(0, min(bottom, img.height))
    return img.crop((left, top, right, bottom))


def no_long_repeats(s: str, threshold: int) -> bool:
    """
    Returns False if the string `s` contains more than `threshold`
    identical characters in a row, otherwise True.
    """
    pattern = r'(.)\1{' + str(threshold) + ',}'
    return re.search(pattern, s) is None


class PostOcrEnrichmentElement(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: Union[DocItem, TableCell, RichTableCell, GraphCell]
    image: list[
        Image.Image
    ]  # Needs to be an a list of images for multi-provenance elements


class PostOcrEnrichmentPipelineOptions(ConvertPipelineOptions):
    api_options: PictureDescriptionApiOptions


class PostOcrEnrichmentPipeline(SimplePipeline):
    def __init__(self, pipeline_options: PostOcrEnrichmentPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: PostOcrEnrichmentPipelineOptions

        self.enrichment_pipe = [
            PostOcrApiEnrichmentModel(
                enabled=True,
                enable_remote_services=True,
                artifacts_path=None,
                options=self.pipeline_options.api_options,
                accelerator_options=AcceleratorOptions(),
            )
        ]

    @classmethod
    def get_default_options(cls) -> PostOcrEnrichmentPipelineOptions:
        return PostOcrEnrichmentPipelineOptions()

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


class PostOcrApiEnrichmentModel(
    GenericEnrichmentModel[PostOcrEnrichmentElement], BaseModelWithOptions
):
    expansion_factor: float = 0.001

    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> Optional[list[PostOcrEnrichmentElement]]:
        if not self.is_processable(doc=conv_res.document, element=element):
            return None

        allowed = (DocItem, TableItem, GraphCell)
        assert isinstance(element, allowed)

        if isinstance(element, (KeyValueItem, FormItem)):
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

                    # cropped_image = safe_crop(conv_res.document.pages[page_ix].image.pil_image, expanded_bbox.as_tuple())
                    is_empty, rem_frac, debug = is_empty_fast_with_lines_pil(cropped_image)
                    if is_empty:
                        if SHOW_EMPTY_CROPS:
                            try:
                                cropped_image.show()
                            except Exception as e:
                                print("Error with image: {}".format(e))
                        print("!!! DETECTED EMPTY FORM ITEM IMAGE CROP !!! {}".format(rem_frac))
                        print(debug)
                    else:
                        result.append(
                            PostOcrEnrichmentElement(item=c, image=[cropped_image])
                        )
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

                            """
                            expanded_bbox = bbox.expand_by_scale(
                                x_scale=self.expansion_factor,
                                y_scale=self.expansion_factor,
                            ).to_top_left_origin(
                                page_height=conv_res.document.pages[
                                    page_ix
                                ].image.size.height
                            )
                            """

                            expanded_bbox = bbox.expand_by_scale(
                                x_scale=0,
                                y_scale=0,
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

                                # cropped_image = safe_crop(conv_res.document.pages[page_ix].image.pil_image, expanded_bbox.as_tuple())
                                is_empty, rem_frac, debug = is_empty_fast_with_lines_pil(cropped_image)
                                if is_empty:
                                    if SHOW_EMPTY_CROPS:
                                        try:
                                            cropped_image.show()
                                        except Exception as e:
                                            print("Error with image: {}".format(e))
                                    print("!!! DETECTED EMPTY TABLE CELL IMAGE CROP !!! {}".format(rem_frac))
                                    print(debug)
                                else:
                                    if SHOW_NONEMPTY_CROPS:
                                        cropped_image.show()
                                    result.append(
                                        PostOcrEnrichmentElement(
                                            item=cell, image=[cropped_image]
                                        )
                                    )
            return result
        else:
            multiple_crops = []
            # Crop the image form the page
            for element_prov in element.prov:
                # Iterate over provenances
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

                if hasattr(element, "text"):
                    if good_bbox:
                        cropped_image = conv_res.document.pages[
                            page_ix
                        ].image.pil_image.crop(expanded_bbox.as_tuple())
                        # cropped_image = safe_crop(conv_res.document.pages[page_ix].image.pil_image, expanded_bbox.as_tuple())

                        is_empty, rem_frac, debug = is_empty_fast_with_lines_pil(cropped_image)
                        if is_empty:
                            if SHOW_EMPTY_CROPS:
                                try:
                                    cropped_image.show()
                                except Exception as e:
                                    print("Error with image: {}".format(e))
                            print("!!! DETECTED EMPTY TEXT IMAGE CROP !!! {}".format(rem_frac))
                            print(debug)
                        else:
                            multiple_crops.append(cropped_image)
                            print("")
                            print("cropped image size: {}".format(cropped_image.size))
                            print(type(element))
                            if hasattr(element, "text"):
                                print("OLD TEXT: {}".format(element.text))
                else:
                    print("Not a text element")
            if len(multiple_crops) > 0:
                # good crops
                return [PostOcrEnrichmentElement(item=element, image=multiple_crops)]
            else:
                # nothing
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
        self.concurrency = 2
        self.expansion_factor = 0.05
        # self.expansion_factor = 0.0
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
            res = api_image_request(
                image=image,
                prompt=self.options.prompt,
                url=self.options.url,
                # timeout=self.options.timeout,
                timeout=30,
                headers=self.options.headers,
                **self.options.params,
            )
            return res[0]

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

            output = clean_html_tags(output).strip()
            output = remove_break_lines(output)
            # The last measure against hallucinations
            # Detect hallucinated string...
            if output.startswith("The first of these"):
                output = ""

            if no_long_repeats(output, 50):
                if VERBOSE:
                    if isinstance(item, (TextItem)):
                        print(f"OLD TEXT: {item.text}")

                # Re-populate text
                if isinstance(item, (TextItem, GraphCell)):
                    if img_ind > 0:
                        # Concat texts across several provenances
                        item.text += " " + output
                        # item.orig += " " + output
                    else:
                        item.text = output
                        # item.orig = output
                elif isinstance(item, (TableCell, RichTableCell)):
                    item.text = output
                elif isinstance(item, PictureItem):
                    pass
                else:
                    raise ValueError(f"Unknown item type: {type(item)}")

                if VERBOSE:
                    if isinstance(item, (TextItem)):
                        print(f"NEW TEXT: {item.text}")

                # Take care of charspans for relevant types
                if isinstance(item, GraphCell):
                    item.prov.charspan = (0, len(item.text))
                elif isinstance(item, TextItem):
                    item.prov[0].charspan = (0, len(item.text))

            yield item


def convert_pdf(pdf_path: Path, out_intermediate_json: Path):
    # Let's prepare a Docling document json with embedded page images
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    # pipeline_options.images_scale = 4.0
    pipeline_options.images_scale = 2.0

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

    if VERBOSE:
        print(
            "Converting PDF to get a Docling document json with embedded page images..."
        )
    conv_result = doc_converter.convert(pdf_path)
    conv_result.document.save_as_json(
        filename=out_intermediate_json, image_mode=ImageRefMode.EMBEDDED
    )
    if PRINT_RESULT_MARKDOWN:
        md1 = conv_result.document.export_to_markdown()
        print("*** ORIGINAL MARKDOWN ***")
        print(md1)


def post_process_json(in_json: Path, out_final_json: Path):
    # Post-Process OCR on top of existing Docling document, per element's bounding box:
    print(f"Post-process all bounding boxes with OCR... {os.path.basename(in_json)}")
    pipeline_options = PostOcrEnrichmentPipelineOptions(
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

    # try:
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.JSON_DOCLING: FormatOption(
                pipeline_cls=PostOcrEnrichmentPipeline,
                pipeline_options=pipeline_options,
                backend=DoclingJSONBackend,
            )
        }
    )
    result = doc_converter.convert(in_json)
    if SHOW_IMAGE:
        result.document.pages[1].image.pil_image.show()
    result.document.save_as_json(out_final_json)
    if PRINT_RESULT_MARKDOWN:
        md = result.document.export_to_markdown()
        print("*** MARKDOWN ***")
        print(md)
    # except:
    #     print("ERROR IN OCR for: {}".format(in_json))


def process_pdf(pdf_path: Path, scratch_dir: Path, out_dir: Path):
    inter_json = scratch_dir / (pdf_path.stem + ".json")
    final_json = out_dir / (pdf_path.stem + ".json")
    inter_json.parent.mkdir(parents=True, exist_ok=True)
    final_json.parent.mkdir(parents=True, exist_ok=True)
    if final_json.exists() and final_json.stat().st_size > 0:
        return  # already done
    convert_pdf(pdf_path, inter_json)
    post_process_json(inter_json, final_json)


def process_json(json_path: Path, out_dir: Path):
    final_json = out_dir / (json_path.stem + ".json")
    final_json.parent.mkdir(parents=True, exist_ok=True)
    if final_json.exists() and final_json.stat().st_size > 0:
        return  # already done
    post_process_json(json_path, final_json)


def filter_jsons_by_ocr_list(jsons, folder):
    """
    jsons: list[Path] - JSON files
    folder: Path - folder containing ocr_documents.txt
    """
    ocr_file = folder / "ocr_documents.txt"

    # If the file doesn't exist, return the list unchanged
    if not ocr_file.exists():
        return jsons

    # Read file names (strip whitespace, ignore empty lines)
    with ocr_file.open("r", encoding="utf-8") as f:
        allowed = {line.strip() for line in f if line.strip()}

    # Keep only JSONs whose stem is in allowed list
    filtered = [p for p in jsons if p.stem in allowed]
    return filtered


def run_jsons(in_path: Path, out_dir: Path):
    if in_path.is_dir():
        jsons = sorted(in_path.glob("*.json"))
        if not jsons:
            raise SystemExit("Folder mode expects one or more .json files")
        # TODO: Look for ocr_documents.txt, in case found, respect only the jsons
        filtered_jsons = filter_jsons_by_ocr_list(jsons, in_path)
        for j in tqdm(filtered_jsons):
            print("")
            print("Processing file...")
            print(j)
            process_json(j, out_dir)
    else:
        raise SystemExit("Invalid --in path")


def main():
    logging.getLogger().setLevel(logging.ERROR)
    p = argparse.ArgumentParser(description="PDF/JSON -> final JSON pipeline")
    p.add_argument(
        "--in",
        dest="in_path",
        default="tests/data/pdf/2305.03393v1-pg9.pdf",
        required=False,
        help="Path to a PDF/JSON file or a folder of JSONs",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        default="scratch/",
        required=False,
        help="Folder for final JSONs (scratch goes inside)",
    )
    args = p.parse_args()

    in_path = Path(args.in_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    scratch_dir = out_dir / "temp"

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    if in_path.is_file():
        if in_path.suffix.lower() == ".pdf":
            process_pdf(in_path, scratch_dir, out_dir)
        elif in_path.suffix.lower() == ".json":
            process_json(in_path, out_dir)
        else:
            raise SystemExit("Single-file mode expects a .pdf or .json")
    else:
        run_jsons(in_path, out_dir)


if __name__ == "__main__":
    main()
