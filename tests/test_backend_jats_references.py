# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the reference list of the JATS backend."""

from io import BytesIO

from docling_core.types.doc import (
    DoclingDocument,
    GroupLabel,
    ListItem,
    Script,
)

from docling.datamodel.base_models import DocumentStream

from .test_backend_jats import get_converter


def convert_jats_ref_list(refs: str) -> DoclingDocument:
    xml = f"""<!DOCTYPE article
PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Publishing DTD v1.3 20210610//EN" "JATS-journalpublishing1-3.dtd">
<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article">
  <front>
    <article-meta>
      <title-group><article-title>References Test</article-title></title-group>
    </article-meta>
  </front>
  <back>
    <ref-list><title>References</title>{refs}</ref-list>
  </back>
</article>
"""
    stream = DocumentStream(name="references-test.nxml", stream=BytesIO(xml.encode()))
    return get_converter().convert(stream).document


def _citation_runs(doc: DoclingDocument) -> list[list[tuple[str, str | None]]]:
    """Return the (text, hyperlink) runs of each reference, in document order.

    A reference is a list item, either with its own text or with an inline
    group of runs as its only child.
    """
    citations: list[list[tuple[str, str | None]]] = []
    for item, _level in doc.iterate_items():
        if not isinstance(item, ListItem):
            continue
        runs = [item]
        if item.children:
            (group,) = [child.resolve(doc) for child in item.children]
            assert group.label == GroupLabel.INLINE
            runs = [child.resolve(doc) for child in group.children]
        citations.append(
            [
                (run.text, str(run.hyperlink) if run.hyperlink is not None else None)
                for run in runs
            ]
        )
    return citations


def test_jats_mixed_citation_keeps_external_link_target():
    doc = convert_jats_ref_list(
        '<ref id="r1"><label>1</label><mixed-citation publication-type="other">'
        "BRAVO-CITATION. A work. 2020.\n"
        '<ext-link ext-link-type="uri" xlink:href="https://example.org/c2-citation">'
        "https://example.org/c2-citation</ext-link></mixed-citation></ref>"
    )

    assert _citation_runs(doc) == [
        [
            ("BRAVO-CITATION. A work. 2020.", None),
            ("https://example.org/c2-citation", "https://example.org/c2-citation"),
        ]
    ]
    assert (
        "- BRAVO-CITATION. A work. 2020. "
        "[https://example.org/c2-citation](https://example.org/c2-citation)"
    ) in doc.export_to_markdown()


def test_jats_mixed_citation_keeps_link_target_that_differs_from_its_text():
    doc = convert_jats_ref_list(
        '<ref id="r1"><mixed-citation>Doe J. Survey data. '
        '<ext-link ext-link-type="uri" xlink:href="https://doi.org/10.5555/survey">'
        "doi:10.5555/survey</ext-link>. Accessed 2020.</mixed-citation></ref>"
    )

    assert _citation_runs(doc) == [
        [
            ("Doe J. Survey data.", None),
            ("doi:10.5555/survey", "https://doi.org/10.5555/survey"),
            (". Accessed 2020.", None),
        ]
    ]


def test_jats_mixed_citation_with_one_run_stays_a_single_list_item():
    doc = convert_jats_ref_list(
        '<ref id="r1"><mixed-citation>Roe R. Plain work. 2019.</mixed-citation></ref>'
        '<ref id="r2"><mixed-citation><ext-link ext-link-type="uri" '
        'xlink:href="https://example.org/data">Example data set</ext-link>'
        "</mixed-citation></ref>"
    )

    assert not [group for group in doc.groups if group.label == GroupLabel.INLINE]
    assert _citation_runs(doc) == [
        [("Roe R. Plain work. 2019.", None)],
        [("Example data set", "https://example.org/data")],
    ]


def test_jats_mixed_citation_keeps_inline_styling():
    doc = convert_jats_ref_list(
        '<ref id="r1"><mixed-citation>Doe J. Growth of '
        "<italic>Escherichia coli</italic>. <source>Microbiology</source>. "
        "2020;<bold>12</bold>: 1-9.</mixed-citation></ref>"
    )

    (group,) = [group for group in doc.groups if group.label == GroupLabel.INLINE]
    runs = [child.resolve(doc) for child in group.children]
    assert [
        (
            run.text,
            (run.formatting.bold, run.formatting.italic, run.formatting.script)
            if run.formatting is not None
            else None,
        )
        for run in runs
    ] == [
        ("Doe J. Growth of", None),
        ("Escherichia coli", (False, True, Script.BASELINE)),
        (". Microbiology. 2020;", None),
        ("12", (True, False, Script.BASELINE)),
        (": 1-9.", None),
    ]
