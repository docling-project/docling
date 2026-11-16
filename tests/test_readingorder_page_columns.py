# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import PurePath

from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.document import ContentLayer

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    InputFormat,
    Page,
    TextElement,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)


def test_parcel_column_precedes_full_width_body() -> None:
    # Simplified layout of NYSample.pdf from issue #1916. The owner column
    # extends below the parcel column. Both columns precede the wide body.
    rows = [
        ("County of Steuben", DocItemLabel.PAGE_HEADER, (230, 24, 311, 46)),
        ("Town Hall", DocItemLabel.TEXT, (50, 61, 246, 95)),
        ("ASSESSMENT NOTIFICATION", DocItemLabel.SECTION_HEADER, (63, 110, 228, 120)),
        ("OWNER INFORMATION", DocItemLabel.SECTION_HEADER, (64, 130, 174, 140)),
        ("PARCEL INFORMATION Tower Site", DocItemLabel.TEXT, (421, 53, 545, 91)),
        ("You are hereby notified.", DocItemLabel.TEXT, (40, 237, 537, 271)),
        (
            "Your tax liability will be affected.",
            DocItemLabel.TEXT,
            (40, 402, 539, 627),
        ),
        ("Assessor", DocItemLabel.TEXT, (371, 630, 445, 651)),
    ]
    elements = [
        TextElement(
            id=index,
            page_no=1,
            label=label,
            text=text,
            cluster=Cluster(
                id=index,
                label=label,
                bbox=BoundingBox(l=left, t=top, r=right, b=bottom),
            ),
        )
        for index, (text, label, (left, top, right, bottom)) in enumerate(rows)
    ]
    conv_res = ConversionResult(
        input=InputDocument.model_construct(
            file=PurePath("NYSample.pdf"),
            document_hash="0" * 64,
            valid=True,
            format=InputFormat.PDF,
        ),
        pages=[Page(page_no=1, size=Size(width=612, height=792))],
        assembled=AssembledUnit(elements=elements, body=elements),
    )

    doc = ReadingOrderModel(ReadingOrderOptions())(conv_res)

    expected = [row[0] for row in rows[1:]]
    body_text = [item.text for item, _ in doc.iterate_items()]
    assert body_text == expected
    markdown = doc.export_to_markdown()
    assert markdown.index("Tower Site") < markdown.index("You are hereby notified.")
    assert "County of Steuben" not in markdown
    assert "County of Steuben" in doc.export_to_markdown(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    )
