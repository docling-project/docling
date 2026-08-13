import os
from pathlib import Path

from docling_core.types.doc import CoordOrigin, DocItemLabel
from docling_core.types.doc.page import TextCellUnit

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import NativePdfPipelineOptions
from docling.datamodel.settings import PageRange
from docling.document_converter import DocumentConverter, NativePdfFormatOption

TEXT_PDF = Path("tests/data/pdf/sources/2305.03393v1-pg9.pdf")
PICTURE_PDF = Path("tests/data/pdf/sources/picture_classification.pdf")
MULTIPAGE_PDF = Path("tests/data/pdf/sources/normal_4pages.pdf")


def _convert(
    source: Path, page_range: PageRange | None = None, **options
) -> ConversionResult:
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: NativePdfFormatOption(
                pipeline_options=NativePdfPipelineOptions(**options)
            )
        }
    )
    kwargs = {} if page_range is None else {"page_range": page_range}
    return converter.convert(source, **kwargs)


def test_native_pipeline_emits_one_text_item_per_line_cell():
    conv_res = _convert(TEXT_PDF)

    assert conv_res.status == ConversionStatus.SUCCESS
    doc = conv_res.document
    assert doc.name == TEXT_PDF.stem
    assert list(doc.pages) == [1]

    line_cells = [
        cell for cell in conv_res.pages[0].parsed_page.textline_cells if cell.text
    ]
    assert len(doc.texts) == len(line_cells)
    assert {text.label for text in doc.texts} == {DocItemLabel.TEXT}
    assert "Optimized Table Tokenization for Table Structure Recognition" in (
        doc.export_to_markdown()
    )


def test_native_pipeline_text_provenance_matches_the_page():
    conv_res = _convert(TEXT_PDF)

    page_size = conv_res.document.pages[1].size
    for text in conv_res.document.texts:
        assert len(text.prov) == 1
        prov = text.prov[0]
        assert prov.page_no == 1
        # Provenance boxes are expressed bottom-left, inside the page.
        assert prov.bbox.coord_origin == CoordOrigin.BOTTOMLEFT
        assert 0 <= prov.bbox.l <= prov.bbox.r <= page_size.width
        assert 0 <= prov.bbox.b <= prov.bbox.t <= page_size.height
        assert prov.charspan == (0, len(text.text))


def test_native_pipeline_word_unit_is_finer_than_line_unit():
    lines = _convert(TEXT_PDF).document
    words = _convert(TEXT_PDF, text_cell_unit=TextCellUnit.WORD).document

    assert len(words.texts) > len(lines.texts)
    assert all(" " not in text.text for text in words.texts)


def test_native_pipeline_extracts_native_bitmaps_as_pictures():
    doc = _convert(PICTURE_PDF).document

    assert len(doc.pictures) > 0
    assert all(picture.image is not None for picture in doc.pictures)
    assert all(len(picture.prov) == 1 for picture in doc.pictures)


def test_native_pipeline_can_skip_picture_images():
    doc = _convert(PICTURE_PDF, generate_picture_images=False).document

    # The pictures are still located on the page, they just carry no image.
    assert len(doc.pictures) > 0
    assert all(picture.image is None for picture in doc.pictures)


def test_native_pipeline_renders_page_images_at_the_requested_scale():
    doc = _convert(TEXT_PDF, images_scale=2.0).document

    page = doc.pages[1]
    assert page.image is not None
    assert page.image.dpi == 144
    assert page.image.pil_image.width == round(page.size.width * 2.0)


def test_native_pipeline_without_page_images_parses_only():
    conv_res = _convert(TEXT_PDF, generate_page_images=False)

    assert conv_res.status == ConversionStatus.SUCCESS
    assert conv_res.document.pages[1].image is None
    assert len(conv_res.document.texts) > 0


def test_native_format_option_parses_with_all_but_one_cpu_thread():
    format_option = NativePdfFormatOption()

    expected = max(1, (os.cpu_count() or 2) - 1)
    assert format_option.pipeline_options.parser_threads == expected
    assert format_option.backend_options.parser_threads == expected

    # The parser threads are independent from the (unused) inference threads.
    format_option = NativePdfFormatOption(
        pipeline_options=NativePdfPipelineOptions(
            parser_threads=3, accelerator_options=AcceleratorOptions(num_threads=7)
        )
    )
    assert format_option.backend_options.parser_threads == 3


def test_native_pipeline_restores_document_order(monkeypatch):
    """The threaded backend yields pages in completion order, not page order."""
    original_iter_pages = ThreadedDoclingParseDocumentBackend.iter_pages

    def _reversed_iter_pages(self):
        yield from reversed(list(original_iter_pages(self)))

    monkeypatch.setattr(
        ThreadedDoclingParseDocumentBackend, "iter_pages", _reversed_iter_pages
    )

    conv_res = _convert(MULTIPAGE_PDF)

    assert [page.page_no for page in conv_res.pages] == [1, 2, 3, 4]
    doc = conv_res.document
    assert list(doc.pages) == [1, 2, 3, 4]
    page_nos = [text.prov[0].page_no for text in doc.texts]
    assert page_nos == sorted(page_nos)


def test_native_pipeline_honors_the_page_range():
    doc = _convert(PICTURE_PDF, page_range=(2, 2)).document

    assert list(doc.pages) == [2]
    assert all(text.prov[0].page_no == 2 for text in doc.texts)
