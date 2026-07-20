import logging
import re
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from bs4 import BeautifulSoup, Tag
from docling_core.types.doc import DocItemLabel, TableData

from docling.datamodel.base_models import Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VlmTableStructureOptions
from docling.models.base_model import BasePageModel
from docling.utils.api_image_request import api_image_request
from docling.utils.llm_cache import in_current_context
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

CHANDRA_TABLE_PROMPT_WITH_OCR_TEMPLATE = """
OCR this image to HTML table. Extract only the table region and output table structure.
Bboxes are normalized to a 0-{bbox_scale} coordinate space.

In addition to the image, you are also provided with OCR text information extracted from the same page.
Each OCR item includes recognized text and its bounding box in the same normalized 0-{bbox_scale} coordinate space.

<ocr_info>
{ocr_info}
</ocr_info>

You MUST use BOTH:
- the visual information from the image
- the provided OCR text and bounding boxes

to accurately reconstruct the table content and structure.

The OCR text should be used as the primary source of textual content.
The image should be used to resolve table structure, merged cells, reading order, and ambiguous cases.

Only use these tags ['table', 'tr', 'td', 'th', 'thead', 'tbody', 'caption', 'p', 'span', 'br', 'b', 'i', 'u', 'sup', 'sub'] and these attributes ['colspan', 'rowspan', 'style', 'align'].

Guidelines:
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, subscripts/superscripts, and special characters.
* Text: join lines properly into cell content.
* Use the simplest possible HTML structure that accurately represents the table.
* Output only table HTML.
""".strip()

CHANDRA_TABLE_PROMPT_NO_OCR_TEMPLATE = """
OCR this image to HTML table. Extract only the table region and output table structure.
Bboxes are normalized to a 0-{bbox_scale} coordinate space.

Only use these tags ['table', 'tr', 'td', 'th', 'thead', 'tbody', 'caption', 'p', 'span', 'br', 'b', 'i', 'u', 'sup', 'sub'] and these attributes ['colspan', 'rowspan', 'style', 'align'].

Guidelines:
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, subscripts/superscripts, and special characters.
* Text: join lines properly into cell content.
* Use the simplest possible HTML structure that accurately represents the table.
* Output only table HTML.
""".strip()

class GenosVlmTableStructureModel(BasePageModel):
    """Table structure model using an external VLM API.

    Sends cropped table images to an OpenAI-compatible vision API and parses
    the returned HTML <table> into docling TableCell structures.
    """

    def __init__(
        self,
        enabled: bool,
        options: VlmTableStructureOptions,
    ):
        self.enabled = enabled
        self.options = options
        self.scale = options.scale

    @staticmethod
    def _extract_table_html(text: str) -> Optional[str]:
        """Extract the first table block from the API response.

        Uses HTML parsing first so nested tables are captured as a complete
        outer table instead of being truncated at the first closing tag.
        """
        soup = BeautifulSoup(text, "html.parser")
        first_table = soup.find("table")
        if isinstance(first_table, Tag):
            return str(first_table)

        # Fallback for non-HTML-like responses.
        match = re.search(r"<table[\s\S]*?</table>", text, re.IGNORECASE)
        if match:
            return match.group(0)
        return None

    @staticmethod
    def _parse_html_to_table_data(html: str) -> Optional[TableData]:
        """Parse an HTML <table> string into TableData using the same logic
        as GenosVlmHTMLDocumentBackend.parse_table_data."""
        from docling.backend.genos_vlm_html_backend import (
            GenosVlmHTMLDocumentBackend,
        )

        soup = BeautifulSoup(html, "html.parser")
        table_tag = soup.find("table")
        if table_tag is None or not isinstance(table_tag, Tag):
            return None

        # Backend parser currently rejects nested tables.
        # Flatten nested tables into plain text so we can still parse structure.
        nested_count = 0
        nested_table = table_tag.find("table")
        while isinstance(nested_table, Tag):
            nested_count += 1
            replacement_text = nested_table.get_text(" ", strip=True)
            if replacement_text:
                nested_table.replace_with(replacement_text)
            else:
                nested_table.decompose()
            nested_table = table_tag.find("table")

        if nested_count > 0:
            _log.debug(f"Flattened {nested_count} nested table(s) before parsing.")

        return GenosVlmHTMLDocumentBackend.parse_table_data(table_tag)

    @staticmethod
    def _normalize_coord(value: float, span: float, bbox_scale: int) -> int:
        if span <= 0:
            return 0
        normalized = int(round((value / span) * bbox_scale))
        return max(0, min(bbox_scale, normalized))

    def _build_table_ocr_info(self, page: Page, table_cluster) -> Optional[str]:
        """Build OCR payload text for the prompt from OCR cells in the table cluster."""
        if not self.options.use_ocr_in_prompt:
            return None

        if page.size is None:
            return None

        bbox_scale = max(1, int(self.options.prompt_bbox_scale))

        table_bbox = table_cluster.bbox.to_top_left_origin(page_height=page.size.height)
        table_w = table_bbox.r - table_bbox.l
        table_h = table_bbox.b - table_bbox.t
        if table_w <= 0 or table_h <= 0:
            return None

        ocr_items = []
        for cell in table_cluster.cells:
            if not cell.from_ocr:
                continue

            text = cell.text.strip()
            if not text:
                continue

            cell_bbox = cell.rect.to_bounding_box().to_top_left_origin(
                page_height=page.size.height
            )

            # Keep OCR boxes within the table crop coordinate frame.
            l = max(cell_bbox.l, table_bbox.l)
            t = max(cell_bbox.t, table_bbox.t)
            r = min(cell_bbox.r, table_bbox.r)
            b = min(cell_bbox.b, table_bbox.b)
            if r <= l or b <= t:
                continue

            ocr_items.append(
                {
                    "text": text,
                    "bbox": [
                        self._normalize_coord(l - table_bbox.l, table_w, bbox_scale),
                        self._normalize_coord(t - table_bbox.t, table_h, bbox_scale),
                        self._normalize_coord(r - table_bbox.l, table_w, bbox_scale),
                        self._normalize_coord(b - table_bbox.t, table_h, bbox_scale),
                    ],
                }
            )

        if not ocr_items:
            return None

        return "\n".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in ocr_items
        )

    def _build_table_prompt(self, ocr_info: Optional[str]) -> str:
        bbox_scale = max(1, int(self.options.prompt_bbox_scale))
        if ocr_info:
            prompt = CHANDRA_TABLE_PROMPT_WITH_OCR_TEMPLATE.format(
                bbox_scale=bbox_scale,
                ocr_info=ocr_info,
            )
        else:
            prompt = CHANDRA_TABLE_PROMPT_NO_OCR_TEMPLATE.format(
                bbox_scale=bbox_scale
            )

        # Keep backwards compatibility for existing custom instruction usage.
        custom_prompt = self.options.prompt.strip()
        if custom_prompt:
            prompt = f"{prompt}\n\nAdditional instructions:\n{custom_prompt}"

        return prompt

    def _process_single_table(
        self,
        page: Page,
        table_cluster,
        ocr_info: Optional[str] = None,
    ) -> Optional[Table]:
        """Process a single table cluster: crop, call API, parse HTML."""
        try:
            crop_image = page.get_image(scale=self.scale, cropbox=table_cluster.bbox)
            if crop_image is None:
                _log.warning(
                    f"Could not get cropped image for table cluster {table_cluster.id}"
                )
                return None

            # Build effective headers
            effective_headers = dict(self.options.headers)
            if self.options.api_key:
                effective_headers["Authorization"] = (
                    f"Bearer {self.options.api_key}"
                )

            # Build effective params
            effective_params = dict(self.options.params)
            if self.options.model:
                effective_params["model"] = self.options.model
            effective_params["temperature"] = self.options.temperature

            prompt = self._build_table_prompt(ocr_info=ocr_info)

            response_text = api_image_request(
                image=crop_image,
                prompt=prompt,
                url=self.options.url,
                timeout=self.options.timeout,
                headers=effective_headers or None,
                **effective_params,
            )

            table_html = self._extract_table_html(response_text)
            if table_html is None:
                _log.warning(
                    f"No <table> found in API response for cluster {table_cluster.id}. "
                    f"Response: {response_text[:200]}"
                )
                return None

            table_data = self._parse_html_to_table_data(table_html)
            if table_data is None:
                _log.warning(
                    f"Failed to parse table HTML for cluster {table_cluster.id}"
                )
                return None

            tbl = Table(
                otsl_seq=[],
                table_cells=table_data.table_cells,
                num_rows=table_data.num_rows,
                num_cols=table_data.num_cols,
                id=table_cluster.id,
                page_no=page.page_no,
                cluster=table_cluster,
                label=table_cluster.label,
            )
            return tbl

        except Exception:
            _log.exception(
                f"Error processing table cluster {table_cluster.id} via VLM API"
            )
            return None

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "table_structure"):
                    assert page.predictions.layout is not None
                    assert page.size is not None

                    page.predictions.tablestructure = TableStructurePrediction()

                    table_clusters = [
                        cluster
                        for cluster in page.predictions.layout.clusters
                        if cluster.label
                        in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
                    ]

                    if not table_clusters:
                        yield page
                        continue

                    concurrency = max(1, self.options.concurrency)
                    ocr_info_by_cluster = {
                        cluster.id: self._build_table_ocr_info(page, cluster)
                        for cluster in table_clusters
                    }

                    if concurrency == 1:
                        for cluster in table_clusters:
                            tbl = self._process_single_table(
                                page,
                                cluster,
                                ocr_info=ocr_info_by_cluster.get(cluster.id),
                            )
                            if tbl is not None:
                                page.predictions.tablestructure.table_map[
                                    cluster.id
                                ] = tbl
                    else:
                        with ThreadPoolExecutor(
                            max_workers=concurrency
                        ) as executor:
                            futures = {
                                executor.submit(
                                    # #329: 워커 스레드에도 llm_cache 컨텍스트 전파
                                    in_current_context(self._process_single_table),
                                    page,
                                    cluster,
                                    ocr_info_by_cluster.get(cluster.id),
                                ): cluster
                                for cluster in table_clusters
                            }
                            for future in as_completed(futures):
                                cluster = futures[future]
                                tbl = future.result()
                                if tbl is not None:
                                    page.predictions.tablestructure.table_map[
                                        cluster.id
                                    ] = tbl

                yield page
