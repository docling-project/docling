# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tagged-PDF structure as the source of layout (ISO 32000-2, 14.7; WTPDF 1.0).

``tagged_ua2_sample.pdf`` is a one-page PDF/UA-2 file (veraPDF 1.30 confirms
full conformance): a Document in the PDF 2.0 namespace holding an H1, two P
elements and an L with three LI, each Lbl + LBody, plus a pagination
artifact. The tests check that the stage derives the same elements from the
tags that the layout model would have to guess, and that heading levels and
artifacts come from the tree.
"""

from pathlib import Path

from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.document import SectionHeaderItem
from docling_parse.pdf_parser import (
    PdfMarkedContentRef,
    PdfStructure,
    PdfStructureElement,
)

from docling.datamodel.base_models import (
    Cluster,
    InputFormat,
    LayoutPrediction,
    Page,
    TaggedTextCell,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.stages.tagged_structure.tagged_structure_model import (
    TaggedStructureModel,
)

TAGGED_PDF = Path("tests/data/pdf/sources/tagged_ua2_sample.pdf")


def _convert(mode: str):
    options = PdfPipelineOptions(
        do_ocr=False, do_table_structure=False, tagged_structure=mode
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    return converter.convert(TAGGED_PDF)


def test_structure_elements_become_clusters_with_levels_alt_and_furniture():
    element = lambda kind, kids, **extra: PdfStructureElement(  # noqa: E731
        id=kind, type=kind, kids=kids, **extra
    )
    mc = lambda mcid: PdfMarkedContentRef(page=0, mcid=mcid)  # noqa: E731
    structure = PdfStructure(
        marked=True,
        role_map={"/Heading": "/H2"},
        elements=[
            element(
                "/Document",
                [
                    element("/Heading", [mc(0)]),
                    element("/P", [mc(1), element("/Span", [mc(2)])]),
                    element(
                        "/Figure",
                        [],
                        page=0,
                        alt="A grey square",
                        attributes={"/Layout": {"/BBox": [20, 20, 80, 60]}},
                    ),
                    element(
                        "/L",
                        [
                            element(
                                "/LI",
                                [element("/Lbl", [mc(3)]), element("/LBody", [mc(4)])],
                            )
                        ],
                    ),
                ],
            )
        ],
    )
    cells = [
        TaggedTextCell(text="Title", bbox=BoundingBox(l=10, t=10, r=60, b=20), mcid=0),
        TaggedTextCell(text="Body", bbox=BoundingBox(l=10, t=30, r=40, b=40), mcid=1),
        TaggedTextCell(text="span", bbox=BoundingBox(l=40, t=30, r=70, b=40), mcid=2),
        TaggedTextCell(text="1.", bbox=BoundingBox(l=10, t=50, r=15, b=60), mcid=3),
        TaggedTextCell(text="item", bbox=BoundingBox(l=15, t=50, r=50, b=60), mcid=4),
        TaggedTextCell(
            text="Page 1",
            bbox=BoundingBox(l=40, t=90, r=60, b=95),
            artifact_type="/Pagination",
            artifact_subtype="/Footer",
        ),
    ]
    page = Page(page_no=1, size=Size(width=100, height=100))

    clusters, prediction = TaggedStructureModel(mode="prefer")._clusters_from_structure(
        structure, page, cells
    )

    assert [c.label for c in clusters] == [
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.TEXT,
        DocItemLabel.PICTURE,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.PAGE_FOOTER,
    ]
    assert all(c.confidence == 1.0 for c in clusters)
    # inline Span content is absorbed into its paragraph's box
    assert clusters[1].bbox == BoundingBox(l=10, t=30, r=70, b=40)
    # role-mapped heading keeps its explicit level; alt text rides along
    assert prediction.heading_levels == {0: 2}
    assert prediction.alt_texts == {2: "A grey square"}
    # a graphics-only figure is placed from its Layout /BBox (bottom-left in the file)
    assert clusters[2].bbox == BoundingBox(l=20, t=40, r=80, b=80)


def test_prefer_keeps_model_clusters_the_tags_did_not_cover():
    tagged = [
        Cluster(id=0, label=DocItemLabel.TEXT, bbox=BoundingBox(l=0, t=0, r=50, b=50))
    ]
    predicted = [
        Cluster(id=7, label=DocItemLabel.TEXT, bbox=BoundingBox(l=5, t=5, r=45, b=45)),
        Cluster(
            id=9, label=DocItemLabel.PICTURE, bbox=BoundingBox(l=60, t=60, r=90, b=90)
        ),
    ]
    merged = TaggedStructureModel._merge_uncovered(tagged, predicted)
    assert [(c.id, c.label) for c in merged] == [
        (0, DocItemLabel.TEXT),
        (1, DocItemLabel.PICTURE),
    ]


def test_pipeline_derives_layout_from_the_tags():
    result = _convert("prefer")
    page = result.pages[0]
    assert page.predictions.tagged_structure is not None
    assert page.predictions.tagged_structure.used
    assert page.predictions.layout is not None
    labels = [c.label for c in page.predictions.layout.clusters]
    assert labels.count(DocItemLabel.SECTION_HEADER) == 1
    assert labels.count(DocItemLabel.LIST_ITEM) == 3

    doc = result.document
    headings = [t for t in doc.texts if isinstance(t, SectionHeaderItem)]
    assert [h.text for h in headings] == ["Accessible Documents Report"]
    assert headings[0].level == 1
    list_items = [t.text for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]
    assert len(list_items) == 3
    paragraphs = [t.text for t in doc.texts if t.label == DocItemLabel.TEXT]
    assert len(paragraphs) == 2
    assert paragraphs[0].startswith("This is the first paragraph")


def test_off_leaves_the_layout_model_in_charge():
    result = _convert("off")
    page = result.pages[0]
    assert page.predictions.tagged_structure is None
    assert page.predictions.layout is not None
    assert all(c.confidence < 1.0 for c in page.predictions.layout.clusters)


def test_pages_without_a_backend_pass_through_untouched():
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.predictions.layout = LayoutPrediction(
        clusters=[
            Cluster(id=0, label=DocItemLabel.TEXT, bbox=BoundingBox(l=0, t=0, r=1, b=1))
        ]
    )
    # no backend: the stage cannot read tags and must not touch the page
    assert list(TaggedStructureModel(mode="require")(None, [page])) == [page]  # type: ignore[arg-type]
    assert page.predictions.layout.clusters
