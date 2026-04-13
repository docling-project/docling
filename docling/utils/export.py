import copy
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import DoclingDocument, PictureItem
from docling_core.types.legacy_doc.base import BaseCell, BaseText, Ref, Table

from docling.datamodel.document import ConversionResult, Page

_log = logging.getLogger(__name__)


def export_to_markdown(
    doc: DoclingDocument,
    image_dir: Optional[Path] = None,
    image_path_prefix: str = "",
    image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
    **kwargs: Any,
) -> str:
    """Export a DoclingDocument to Markdown, optionally saving images to disk.

    This is a convenience wrapper around :meth:`DoclingDocument.export_to_markdown`
    that adds two extra parameters for automatic image saving.

    When ``image_dir`` is provided together with
    ``image_mode=ImageRefMode.REFERENCED``, all embedded picture images are
    automatically saved to *image_dir* using sequential filenames
    (``image_001.png``, ``image_002.png``, …).  The markdown image references
    are rendered as ``![Image](<image_path_prefix>image_00N.png)``.

    Without ``image_dir`` the function delegates directly to
    :meth:`DoclingDocument.export_to_markdown`, so existing call-sites are
    unaffected.

    Args:
        doc: The :class:`DoclingDocument` to serialize.
        image_dir: Directory where images should be saved.  Created
            automatically if it does not exist.  Only used when
            ``image_mode=ImageRefMode.REFERENCED``.
        image_path_prefix: Prefix prepended to each image filename in the
            Markdown output (e.g. ``"images/"``).  Defaults to ``""``.
        image_mode: How images are included in the output.  Defaults to
            :attr:`ImageRefMode.PLACEHOLDER`.  Pass
            :attr:`ImageRefMode.REFERENCED` together with *image_dir* to
            get external image references.
        **kwargs: All remaining keyword arguments are forwarded verbatim to
            :meth:`DoclingDocument.export_to_markdown`.

    Returns:
        The Markdown string.

    Example::

        from pathlib import Path
        from docling_core.types.doc.base import ImageRefMode
        from docling.utils.export import export_to_markdown

        md = export_to_markdown(
            result.document,
            image_mode=ImageRefMode.REFERENCED,
            image_dir=Path("./output/images"),
            image_path_prefix="images/",
        )
    """
    if image_dir is None or image_mode != ImageRefMode.REFERENCED:
        # Fast path: nothing special to do — delegate directly.
        return doc.export_to_markdown(image_mode=image_mode, **kwargs)

    # --- Save images and rewrite URIs in a deep copy so the original is untouched ---
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    doc_copy: DoclingDocument = copy.deepcopy(doc)

    img_count = 0
    for item, _ in doc_copy.iterate_items(with_groups=False):
        if not isinstance(item, PictureItem):
            continue

        # Use the copy's item to get the image, but resolve pixel data from the
        # *original* document's page images (still referenced inside the deep copy).
        img = item.get_image(doc=doc_copy)
        if img is None:
            img_count += 1
            continue

        filename = f"image_{img_count + 1:03d}.png"
        save_path = image_dir / filename
        img.save(save_path)

        # Build the URI that will appear in the Markdown output.
        uri_str = f"{image_path_prefix}{filename}"

        # Ensure item.image exists so we can set its uri.
        if item.image is None:
            from docling_core.types.doc.document import ImageRef  # local import to avoid cycles
            scale = img.size[0] / item.prov[0].bbox.width if item.prov else 1.0
            item.image = ImageRef.from_pil(image=img, dpi=round(72 * scale))

        item.image.uri = Path(uri_str)  # type: ignore[assignment]

        img_count += 1

    return doc_copy.export_to_markdown(image_mode=image_mode, **kwargs)


def generate_multimodal_pages(
    doc_result: ConversionResult,
) -> Iterable[Tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]], Page]]:
    label_to_doclaynet = {
        "title": "title",
        "table-of-contents": "document_index",
        "subtitle-level-1": "section_header",
        "checkbox-selected": "checkbox_selected",
        "checkbox-unselected": "checkbox_unselected",
        "caption": "caption",
        "page-header": "page_header",
        "page-footer": "page_footer",
        "footnote": "footnote",
        "table": "table",
        "formula": "formula",
        "list-item": "list_item",
        "code": "code",
        "figure": "picture",
        "picture": "picture",
        "reference": "text",
        "paragraph": "text",
        "text": "text",
    }

    content_text = ""
    page_no = 0
    start_ix = 0
    end_ix = 0
    doc_items: List[Tuple[int, Union[BaseCell, BaseText]]] = []

    doc = doc_result.legacy_document

    def _process_page_segments(doc_items: list[Tuple[int, BaseCell]], page: Page):
        segments = []

        for ix, item in doc_items:
            item_type = item.obj_type
            label = label_to_doclaynet.get(item_type, None)

            if label is None or item.prov is None or page.size is None:
                continue

            bbox = BoundingBox.from_tuple(
                tuple(item.prov[0].bbox), origin=CoordOrigin.BOTTOMLEFT
            )
            new_bbox = bbox.to_top_left_origin(page_height=page.size.height).normalized(
                page_size=page.size
            )

            new_segment = {
                "index_in_doc": ix,
                "label": label,
                "text": item.text if item.text is not None else "",
                "bbox": new_bbox.as_tuple(),
                "data": [],
            }

            if isinstance(item, Table):
                table_html = item.export_to_html()
                new_segment["data"].append(
                    {
                        "html_seq": table_html,
                        "otsl_seq": "",
                    }
                )

            segments.append(new_segment)

        return segments

    def _process_page_cells(page: Page):
        cells: List[dict] = []
        if page.size is None:
            return cells
        for cell in page.cells:
            new_bbox = (
                cell.rect.to_bounding_box()
                .to_top_left_origin(page_height=page.size.height)
                .normalized(page_size=page.size)
            )
            is_ocr = cell.from_ocr
            ocr_confidence = cell.confidence
            cells.append(
                {
                    "text": cell.text,
                    "bbox": new_bbox.as_tuple(),
                    "ocr": is_ocr,
                    "ocr_confidence": ocr_confidence,
                }
            )
        return cells

    def _process_page():
        page_ix = page_no - 1
        page = doc_result.pages[page_ix]

        page_cells = _process_page_cells(page=page)
        page_segments = _process_page_segments(doc_items=doc_items, page=page)
        content_md = doc.export_to_markdown(
            main_text_start=start_ix, main_text_stop=end_ix
        )
        # No page-tagging since we only do 1 page at the time
        content_dt = doc.export_to_document_tokens(
            main_text_start=start_ix, main_text_stop=end_ix, add_page_index=False
        )

        return content_text, content_md, content_dt, page_cells, page_segments, page

    if doc.main_text is None:
        return
    for ix, orig_item in enumerate(doc.main_text):
        item = doc._resolve_ref(orig_item) if isinstance(orig_item, Ref) else orig_item
        if item is None or item.prov is None or len(item.prov) == 0:
            _log.debug(f"Skipping item {orig_item}")
            continue

        item_page = item.prov[0].page

        # Page is complete
        if page_no > 0 and item_page > page_no:
            yield _process_page()

            start_ix = ix
            doc_items = []
            content_text = ""

        page_no = item_page
        end_ix = ix
        doc_items.append((ix, item))
        if item.text is not None and item.text != "":
            content_text += item.text + " "

    if len(doc_items) > 0:
        yield _process_page()
