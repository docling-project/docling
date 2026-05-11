import os
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def get_jats_paths():
    directory = Path(os.path.dirname(__file__) + "/data/jats/")
    xml_files = sorted(directory.rglob("*.nxml"))
    return xml_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.XML_JATS])
    return converter


def convert_jats_contribs(contribs: str, affiliations: str = "") -> DoclingDocument:
    xml = f"""<!DOCTYPE article
PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">
<article article-type="research-article">
  <front>
    <article-meta>
      <title-group><article-title>Author Variant Test</article-title></title-group>
      <contrib-group>{contribs}</contrib-group>
      {affiliations}
    </article-meta>
  </front>
</article>
"""
    stream = DocumentStream(
        name="author-variant-test.nxml", stream=BytesIO(xml.encode())
    )
    conv_result: ConversionResult = get_converter().convert(stream)
    return conv_result.document


@pytest.mark.parametrize(
    ("contrib", "expected"),
    [
        (
            """<contrib contrib-type="author"><name><given-names>Jane</given-names><surname>Doe</surname></name></contrib>""",
            "Jane Doe",
        ),
        (
            """<contrib contrib-type="author"><string-name>Jane Q. Doe</string-name></contrib>""",
            "Jane Q. Doe",
        ),
        (
            """<contrib contrib-type="author"><name-alternatives><name><given-names>Jane</given-names><surname>Doe</surname></name><string-name>J. Doe</string-name></name-alternatives></contrib>""",
            "Jane Doe",
        ),
        (
            """<contrib contrib-type="author"><collab-name>Example Working Group</collab-name></contrib>""",
            "Example Working Group",
        ),
        (
            """<contrib contrib-type="author"><collab>Deprecated Working Group</collab></contrib>""",
            "Deprecated Working Group",
        ),
        (
            """<contrib contrib-type="author"><collab-name-alternatives><collab-name>Primary Group</collab-name><collab>Legacy Group</collab></collab-name-alternatives></contrib>""",
            "Primary Group",
        ),
        (
            """<contrib contrib-type="author"><collab-alternatives><collab>Alternative Group</collab></collab-alternatives></contrib>""",
            "Alternative Group",
        ),
        (
            """<contrib contrib-type="author"><anonymous/></contrib>""",
            "Anonymous",
        ),
        (
            """<contrib contrib-type="author"><name><surname>Doe</surname></name></contrib>""",
            "Doe",
        ),
        (
            """<contrib contrib-type="author"><name><given-names>Jane</given-names></name></contrib>""",
            "Jane",
        ),
    ],
)
def test_jats_author_name_variants(contrib: str, expected: str):
    doc = convert_jats_contribs(contrib)

    assert expected in doc.export_to_markdown()


def test_jats_author_affiliations_still_map_from_xref():
    doc = convert_jats_contribs(
        """<contrib contrib-type="author"><name><given-names>Jane</given-names><surname>Doe</surname></name><xref ref-type="aff" rid="aff1">1</xref></contrib>""",
        """<aff id="aff1"><label>1</label><addr-line>Example University</addr-line></aff>""",
    )

    md = doc.export_to_markdown()
    assert "Jane Doe" in md
    assert "Example University" in md


def test_e2e_jats_conversions(use_stream=False):
    jats_paths = get_jats_paths()
    converter = get_converter()

    for jats_path in jats_paths:
        gt_path = (
            jats_path.parent.parent / "groundtruth" / "docling_v2" / jats_path.name
        )
        if use_stream:
            buf = BytesIO(jats_path.open("rb").read())
            stream = DocumentStream(name=jats_path.name, stream=buf)
            conv_result: ConversionResult = converter.convert(stream)
        else:
            conv_result: ConversionResult = converter.convert(jats_path)
        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            "export to md"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), "export to json"


def test_e2e_jats_conversions_stream():
    test_e2e_jats_conversions(use_stream=True)


def test_e2e_jats_conversions_no_stream():
    test_e2e_jats_conversions(use_stream=False)
