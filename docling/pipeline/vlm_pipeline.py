import logging
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union, cast

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    DocItem,
    DoclingDocument,
    ImageRef,
    PictureItem,
    ProvenanceItem,
    TableCell,
    TableData,
    TextItem,
)
from docling_core.types.doc.base import (
    BoundingBox,
    Size,
)
from docling_core.types.doc.document import DocTagsDocument
from lxml import etree
from PIL import Image as PILImage

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.settings import settings
from docling.models.api_vlm_model import ApiVlmModel
from docling.models.vlm_models_inline.hf_transformers_model import (
    HuggingFaceTransformersVlmModel,
)
from docling.models.vlm_models_inline.mlx_model import HuggingFaceMlxModel
from docling.pipeline.base_pipeline import PaginatedPipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class VlmPipeline(PaginatedPipeline):
    def __init__(self, pipeline_options: VlmPipelineOptions):
        super().__init__(pipeline_options)
        self.keep_backend = True

        self.pipeline_options: VlmPipelineOptions

        # force_backend_text = False - use text that is coming from VLM response
        # force_backend_text = True - get text from backend using bounding boxes predicted by SmolDocling doctags
        self.force_backend_text = (
            pipeline_options.force_backend_text
            and pipeline_options.vlm_options.response_format == ResponseFormat.DOCTAGS
        )

        self.keep_images = self.pipeline_options.generate_page_images

        if isinstance(pipeline_options.vlm_options, ApiVlmOptions):
            self.build_pipe = [
                ApiVlmModel(
                    enabled=True,  # must be always enabled for this pipeline to make sense.
                    enable_remote_services=self.pipeline_options.enable_remote_services,
                    vlm_options=cast(ApiVlmOptions, self.pipeline_options.vlm_options),
                ),
            ]
        elif isinstance(self.pipeline_options.vlm_options, InlineVlmOptions):
            vlm_options = cast(InlineVlmOptions, self.pipeline_options.vlm_options)
            if vlm_options.inference_framework == InferenceFramework.MLX:
                self.build_pipe = [
                    HuggingFaceMlxModel(
                        enabled=True,  # must be always enabled for this pipeline to make sense.
                        artifacts_path=self.artifacts_path,
                        accelerator_options=pipeline_options.accelerator_options,
                        vlm_options=vlm_options,
                    ),
                ]
            elif vlm_options.inference_framework == InferenceFramework.TRANSFORMERS:
                self.build_pipe = [
                    HuggingFaceTransformersVlmModel(
                        enabled=True,  # must be always enabled for this pipeline to make sense.
                        artifacts_path=self.artifacts_path,
                        accelerator_options=pipeline_options.accelerator_options,
                        vlm_options=vlm_options,
                    ),
                ]
            elif vlm_options.inference_framework == InferenceFramework.VLLM:
                from docling.models.vlm_models_inline.vllm_model import VllmVlmModel

                self.build_pipe = [
                    VllmVlmModel(
                        enabled=True,  # must be always enabled for this pipeline to make sense.
                        artifacts_path=self.artifacts_path,
                        accelerator_options=pipeline_options.accelerator_options,
                        vlm_options=vlm_options,
                    ),
                ]
            else:
                raise ValueError(
                    f"Could not instantiate the right type of VLM pipeline: {vlm_options.inference_framework}"
                )

        self.enrichment_pipe = [
            # Other models working on `NodeItem` elements in the DoclingDocument
        ]

    def initialize_page(self, conv_res: ConversionResult, page: Page) -> Page:
        with TimeRecorder(conv_res, "page_init"):
            images_scale = self.pipeline_options.images_scale
            if images_scale is not None:
                page._default_image_scale = images_scale
            page._backend = conv_res.input._backend.load_page(page.page_no)  # type: ignore
            if page._backend is not None and page._backend.is_valid():
                page.size = page._backend.get_size()

                if self.force_backend_text:
                    page.parsed_page = page._backend.get_segmented_page()

        return page

    def extract_text_from_backend(
        self, page: Page, bbox: Union[BoundingBox, None]
    ) -> str:
        # Convert bounding box normalized to 0-100 into page coordinates for cropping
        text = ""
        if bbox:
            if page.size:
                if page._backend:
                    text = page._backend.get_text_in_rect(bbox)
        return text

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            if (
                self.pipeline_options.vlm_options.response_format
                == ResponseFormat.DOCTAGS
            ):
                conv_res.document = self._turn_dt_into_doc(conv_res)

            elif self.pipeline_options.vlm_options.response_format in (
                ResponseFormat.MARKDOWN,
                ResponseFormat.ANNOTATED_MARKDOWN,
            ):
                conv_res.document = self._turn_md_into_doc(conv_res)

            elif (
                self.pipeline_options.vlm_options.response_format == ResponseFormat.HTML
            ):
                conv_res.document = self._turn_html_into_doc(conv_res)

            else:
                raise RuntimeError(
                    f"Unsupported VLM response format {self.pipeline_options.vlm_options.response_format}"
                )

            # Generate images of the requested element types
            if self.pipeline_options.generate_picture_images:
                scale = self.pipeline_options.images_scale
                for element, _level in conv_res.document.iterate_items():
                    if not isinstance(element, DocItem) or len(element.prov) == 0:
                        continue
                    if (
                        isinstance(element, PictureItem)
                        and self.pipeline_options.generate_picture_images
                    ):
                        page_ix = element.prov[0].page_no - 1
                        page = conv_res.pages[page_ix]
                        assert page.size is not None
                        assert page.image is not None

                        crop_bbox = (
                            element.prov[0]
                            .bbox.scaled(scale=scale)
                            .to_top_left_origin(page_height=page.size.height * scale)
                        )

                        cropped_im = page.image.crop(crop_bbox.as_tuple())
                        element.image = ImageRef.from_pil(
                            cropped_im, dpi=int(72 * scale)
                        )

        return conv_res

    def _turn_dt_into_doc(self, conv_res) -> DoclingDocument:
        doctags_list = []
        image_list = []
        for page in conv_res.pages:
            predicted_doctags = ""
            img = PILImage.new("RGB", (1, 1), "rgb(255,255,255)")
            if page.predictions.vlm_response:
                predicted_doctags = page.predictions.vlm_response.text
            if page.image:
                img = page.image
            image_list.append(img)
            doctags_list.append(predicted_doctags)

        doctags_list_c = cast(List[Union[Path, str]], doctags_list)
        image_list_c = cast(List[Union[Path, PILImage.Image]], image_list)
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            doctags_list_c, image_list_c
        )
        conv_res.document = DoclingDocument.load_from_doctags(
            doctag_document=doctags_doc
        )

        # If forced backend text, replace model predicted text with backend one
        if page.size:
            if self.force_backend_text:
                scale = self.pipeline_options.images_scale
                for element, _level in conv_res.document.iterate_items():
                    if not isinstance(element, TextItem) or len(element.prov) == 0:
                        continue
                    crop_bbox = (
                        element.prov[0]
                        .bbox.scaled(scale=scale)
                        .to_top_left_origin(page_height=page.size.height * scale)
                    )
                    txt = self.extract_text_from_backend(page, crop_bbox)
                    element.text = txt
                    element.orig = txt

        return conv_res.document

    @staticmethod
    def _parse_table_html_static(html_content: str) -> TableData:
        """Parse HTML table content and create TableData structure.

        Args:
            html_content: HTML string containing <table> element

        Returns:
            TableData with parsed table structure
        """

        # Extract table HTML if wrapped in other content
        table_match = re.search(
            r"<table[^>]*>.*?</table>", html_content, re.DOTALL | re.IGNORECASE
        )
        if not table_match:
            # No table found, return empty table
            return TableData(num_rows=0, num_cols=0, table_cells=[])

        table_html = table_match.group(0)

        try:
            # Parse HTML with lxml
            parser = etree.HTMLParser()
            tree = etree.fromstring(table_html, parser)

            # Find all rows
            rows = tree.xpath(".//tr")
            if not rows:
                return TableData(num_rows=0, num_cols=0, table_cells=[])

            # Calculate grid dimensions
            num_rows = len(rows)
            num_cols = 0

            # First pass: determine number of columns
            for row in rows:
                cells = row.xpath("./td | ./th")
                col_count = 0
                for cell in cells:
                    colspan = int(cell.get("colspan", "1"))
                    col_count += colspan
                num_cols = max(num_cols, col_count)

            # Create grid to track cell positions
            grid: list[list[Union[None | str]]] = [
                [None for _ in range(num_cols)] for _ in range(num_rows)
            ]
            table_data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=[])

            # Second pass: populate cells
            for row_idx, row in enumerate(rows):
                cells = row.xpath("./td | ./th")
                col_idx = 0

                for cell in cells:
                    # Find next available column
                    while col_idx < num_cols and grid[row_idx][col_idx] is not None:
                        col_idx += 1

                    if col_idx >= num_cols:
                        break

                    # Get cell properties
                    text = "".join(cell.itertext()).strip()
                    colspan = int(cell.get("colspan", "1"))
                    rowspan = int(cell.get("rowspan", "1"))
                    is_header = cell.tag.lower() == "th"

                    # Mark grid cells as occupied
                    for r in range(row_idx, min(row_idx + rowspan, num_rows)):
                        for c in range(col_idx, min(col_idx + colspan, num_cols)):
                            grid[r][c] = text

                    # Create table cell
                    table_cell = TableCell(
                        text=text,
                        row_span=rowspan,
                        col_span=colspan,
                        start_row_offset_idx=row_idx,
                        end_row_offset_idx=row_idx + rowspan,
                        start_col_offset_idx=col_idx,
                        end_col_offset_idx=col_idx + colspan,
                        column_header=is_header and row_idx == 0,
                        row_header=is_header and col_idx == 0,
                    )
                    table_data.table_cells.append(table_cell)

                    col_idx += colspan

            return table_data

        except Exception as e:
            _log.warning(f"Failed to parse table HTML: {e}")
            return TableData(num_rows=0, num_cols=0, table_cells=[])

    @staticmethod
    def _collect_annotation_content(
        lines: list[str],
        i: int,
        label_str: str,
        annotation_pattern: str,
        visited_lines: set[int],
    ) -> tuple[str, int]:
        """Collect content for an annotation.

        Args:
            lines: All lines from the document
            i: Current line index (after annotation line)
            label_str: The annotation label (e.g., 'table', 'text')
            annotation_pattern: Regex pattern to match annotations
            visited_lines: Set of already visited line indices

        Returns:
            Tuple of (content string, next line index)
        """
        content_lines = []

        # Special handling for table: extract only <table>...</table>
        if label_str == "table":
            table_started = False
            ii = i
            while ii < len(lines):
                line = lines[ii]
                if "<table" in line.lower():
                    table_started = True
                if table_started:
                    visited_lines.add(ii)
                    content_lines.append(line.rstrip())
                if table_started and "</table>" in line.lower():
                    break
                ii += 1
        else:
            # Original logic for other labels
            while i < len(lines):
                content_line = lines[i].strip()
                if content_line:
                    if re.match(annotation_pattern, content_line):
                        break
                    visited_lines.add(i)
                    content_lines.append(lines[i].rstrip())
                    i += 1
                    if label_str not in ["figure"]:
                        break
                else:
                    i += 1
                    if content_lines:
                        break

        return "\n".join(content_lines), i

    @staticmethod
    def _process_annotation_item(
        label_str: str,
        content: str,
        prov: ProvenanceItem,
        caption_item,
        page_doc: DoclingDocument,
        label_map: dict,
    ) -> None:
        """Process and add a single annotation item to the document.

        Args:
            label_str: The annotation label
            content: The content text
            prov: Provenance information
            caption_item: Optional caption item to link
            page_doc: Document to add item to
            label_map: Mapping of label strings to DocItemLabel
        """
        from docling_core.types.doc import DocItemLabel

        doc_label = label_map.get(label_str, DocItemLabel.TEXT)

        if label_str == "figure":
            page_doc.add_picture(caption=caption_item, prov=prov)
        elif label_str == "table":
            table_data = VlmPipeline._parse_table_html_static(content)
            page_doc.add_table(data=table_data, caption=caption_item, prov=prov)
        elif label_str == "title":
            page_doc.add_title(text=content, prov=prov)
        elif label_str == "sub_title":
            heading_level = 1
            clean_content = content
            if content.startswith("#"):
                hash_count = sum(
                    1
                    for char in content
                    if char == "#"
                    and content.index(char) < len(content)
                    and content[content.index(char)] == "#"
                )
                hash_count = 0
                for char in content:
                    if char == "#":
                        hash_count += 1
                    else:
                        break
                if hash_count > 1:
                    heading_level = hash_count - 1
                clean_content = content[hash_count:].strip()
            page_doc.add_heading(text=clean_content, level=heading_level, prov=prov)
        else:
            page_doc.add_text(label=doc_label, text=content, prov=prov)

    def _parse_annotated_markdown(self, conv_res: ConversionResult) -> DoclingDocument:
        """Parse annotated markdown with label[[x1, y1, x2, y2]] format.

        Labels supported:
        - text: Standard body text
        - title: Main document or section titles
        - sub_title: Secondary headings or sub-headers
        - table: Tabular data
        - table_caption: Descriptive text for tables
        - figure: Image-based elements or diagrams
        - figure_caption: Titles or descriptions for figures/images
        - header / footer: Content at top or bottom margins of pages
        """
        from docling_core.types.doc import (
            DocItemLabel,
            DocumentOrigin,
            TableData,
        )

        # Label mapping
        label_map = {
            "text": DocItemLabel.TEXT,
            "title": DocItemLabel.TITLE,
            "sub_title": DocItemLabel.SECTION_HEADER,
            "table": DocItemLabel.TABLE,
            "table_caption": DocItemLabel.CAPTION,
            "figure": DocItemLabel.PICTURE,
            "figure_caption": DocItemLabel.CAPTION,
            "header": DocItemLabel.PAGE_HEADER,
            "footer": DocItemLabel.PAGE_FOOTER,
        }

        # Pattern to match: label[[x1, y1, x2, y2]]
        annotation_pattern = r"^(\w+)\[\[([0-9., ]+)\]\]\s*$"

        page_docs = []

        for pg_idx, page in enumerate(conv_res.pages):
            # Create a new document for this page
            origin = DocumentOrigin(
                filename=conv_res.input.file.name or "file",
                mimetype="text/markdown",
                binary_hash=0,
            )
            page_doc = DoclingDocument(
                name=conv_res.input.file.stem or "file", origin=origin
            )

            # Get page dimensions
            if page.image is not None:
                pg_width = page.image.width
                pg_height = page.image.height
            else:
                pg_width = 1
                pg_height = 1

            # Add page metadata
            page_doc.add_page(
                page_no=pg_idx + 1,
                size=Size(width=pg_width, height=pg_height),
                image=ImageRef.from_pil(image=page.image, dpi=72)
                if page.image
                else None,
            )

            predicted_text = ""
            if page.predictions.vlm_response:
                predicted_text = page.predictions.vlm_response.text

            # Split into lines and parse - collect all annotations first
            lines = predicted_text.split("\n")
            annotations = []
            i = 0
            visited_lines: set[int] = set()

            while i < len(lines):
                if i in visited_lines:
                    i += 1
                    continue

                line = lines[i].strip()
                match = re.match(annotation_pattern, line)
                if match:
                    label_str = match.group(1)
                    coords_str = match.group(2)

                    try:
                        coords = [float(x.strip()) for x in coords_str.split(",")]
                        if len(coords) == 4:
                            bbox = BoundingBox(
                                l=coords[0], t=coords[1], r=coords[2], b=coords[3]
                            )
                            prov = ProvenanceItem(
                                page_no=pg_idx + 1, bbox=bbox, charspan=[0, 0]
                            )

                            # Get the content (next non-empty line)
                            i += 1
                            content, i = VlmPipeline._collect_annotation_content(
                                lines, i, label_str, annotation_pattern, visited_lines
                            )
                            annotations.append((label_str, content, prov))
                            continue
                    except (ValueError, IndexError):
                        pass
                i += 1

            # Process annotations and link captions that appear AFTER tables/figures
            for idx, (label_str, content, prov) in enumerate(annotations):
                # Check if NEXT annotation is a caption for this table/figure
                # (caption appears AFTER table in the file: table[[...]] then table_caption[[...]])
                caption_item = None
                if label_str in ["table", "figure"] and idx + 1 < len(annotations):
                    next_label, next_content, next_prov = annotations[idx + 1]
                    if (label_str == "table" and next_label == "table_caption") or (
                        label_str == "figure" and next_label == "figure_caption"
                    ):
                        # Create caption item
                        caption_label = label_map.get(next_label, DocItemLabel.CAPTION)
                        caption_item = page_doc.add_text(
                            label=caption_label,
                            text=next_content,
                            prov=next_prov,
                        )

                # Skip if this is a caption that was already processed
                if label_str in ["figure_caption", "table_caption"]:
                    if idx > 0:
                        prev_label = annotations[idx - 1][0]
                        if (label_str == "table_caption" and prev_label == "table") or (
                            label_str == "figure_caption" and prev_label == "figure"
                        ):
                            continue

                # Add the item
                VlmPipeline._process_annotation_item(
                    label_str, content, prov, caption_item, page_doc, label_map
                )

            page_docs.append(page_doc)

        # Return the first page document (or concatenate if multiple pages)
        if len(page_docs) == 1:
            return page_docs[0]
        else:
            final_doc = DoclingDocument.concatenate(docs=page_docs)
            return final_doc

    def _turn_md_into_doc(self, conv_res):
        def _extract_markdown_code(text):
            """
            Extracts text from markdown code blocks (enclosed in triple backticks).
            If no code blocks are found, returns the original text.

            Args:
                text (str): Input text that may contain markdown code blocks

            Returns:
                str: Extracted code if code blocks exist, otherwise original text
            """
            # Regex pattern to match content between triple backticks
            # This handles multiline content and optional language specifier
            pattern = r"^```(?:\w*\n)?(.*?)```(\n)*$"

            # Search with DOTALL flag to match across multiple lines
            mtch = re.search(pattern, text, re.DOTALL)

            if mtch:
                # Return only the content of the first capturing group
                return mtch.group(1)
            else:
                # No code blocks found, return original text
                return text

        page_docs = []

        # Check if we should parse annotations
        if (
            self.pipeline_options.vlm_options.response_format
            == ResponseFormat.ANNOTATED_MARKDOWN
        ):
            # Use specialized annotated markdown parser
            return self._parse_annotated_markdown(conv_res)

        for pg_idx, page in enumerate(conv_res.pages):
            predicted_text = ""
            if page.predictions.vlm_response:
                predicted_text = page.predictions.vlm_response.text + "\n\n"

            predicted_text = _extract_markdown_code(text=predicted_text)

            response_bytes = BytesIO(predicted_text.encode("utf8"))
            out_doc = InputDocument(
                path_or_stream=response_bytes,
                filename=conv_res.input.file.name,
                format=InputFormat.MD,
                backend=MarkdownDocumentBackend,
            )
            backend = MarkdownDocumentBackend(
                in_doc=out_doc,
                path_or_stream=response_bytes,
            )
            page_doc = backend.convert()

            # Modify provenance in place for all items in the page document
            for item, level in page_doc.iterate_items(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers=set(ContentLayer),
            ):
                if isinstance(item, DocItem):
                    item.prov = [
                        ProvenanceItem(
                            page_no=pg_idx + 1,
                            bbox=BoundingBox(
                                t=0.0, b=0.0, l=0.0, r=0.0
                            ),  # FIXME: would be nice not to have to "fake" it
                            charspan=[0, 0],
                        )
                    ]

            # Add page metadata to the page document before concatenation
            if page.image is not None:
                pg_width = page.image.width
                pg_height = page.image.height
            else:
                pg_width = 1
                pg_height = 1

            page_doc.add_page(
                page_no=pg_idx + 1,
                size=Size(width=pg_width, height=pg_height),
                image=ImageRef.from_pil(image=page.image, dpi=72)
                if page.image
                else None,
            )

            page_docs.append(page_doc)

        final_doc = DoclingDocument.concatenate(docs=page_docs)
        return final_doc

    def _turn_html_into_doc(self, conv_res):
        def _extract_html_code(text):
            """
            Extracts text from markdown code blocks (enclosed in triple backticks).
            If no code blocks are found, returns the original text.

            Args:
                text (str): Input text that may contain markdown code blocks

            Returns:
                str: Extracted code if code blocks exist, otherwise original text
            """
            # Regex pattern to match content between triple backticks
            # This handles multiline content and optional language specifier
            pattern = r"^```(?:\w*\n)?(.*?)```(\n)*$"

            # Search with DOTALL flag to match across multiple lines
            mtch = re.search(pattern, text, re.DOTALL)

            if mtch:
                # Return only the content of the first capturing group
                return mtch.group(1)
            else:
                # No code blocks found, return original text
                return text

        page_docs = []

        for pg_idx, page in enumerate(conv_res.pages):
            predicted_text = ""
            if page.predictions.vlm_response:
                predicted_text = page.predictions.vlm_response.text + "\n\n"

            predicted_text = _extract_html_code(text=predicted_text)

            response_bytes = BytesIO(predicted_text.encode("utf8"))
            out_doc = InputDocument(
                path_or_stream=response_bytes,
                filename=conv_res.input.file.name,
                format=InputFormat.HTML,
                backend=HTMLDocumentBackend,
            )
            backend = HTMLDocumentBackend(
                in_doc=out_doc,
                path_or_stream=response_bytes,
            )
            page_doc = backend.convert()

            # Modify provenance in place for all items in the page document
            for item, level in page_doc.iterate_items(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers=set(ContentLayer),
            ):
                if isinstance(item, DocItem):
                    item.prov = [
                        ProvenanceItem(
                            page_no=pg_idx + 1,
                            bbox=BoundingBox(
                                t=0.0, b=0.0, l=0.0, r=0.0
                            ),  # FIXME: would be nice not to have to "fake" it
                            charspan=[0, 0],
                        )
                    ]

            # Add page metadata to the page document before concatenation
            if page.image is not None:
                pg_width = page.image.width
                pg_height = page.image.height
            else:
                pg_width = 1
                pg_height = 1

            page_doc.add_page(
                page_no=pg_idx + 1,
                size=Size(width=pg_width, height=pg_height),
                image=ImageRef.from_pil(image=page.image, dpi=72)
                if page.image
                else None,
            )

            page_docs.append(page_doc)

        # Concatenate all page documents to preserve hierarchy
        final_doc = DoclingDocument.concatenate(docs=page_docs)
        return final_doc

    @classmethod
    def get_default_options(cls) -> VlmPipelineOptions:
        return VlmPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, PdfDocumentBackend)
