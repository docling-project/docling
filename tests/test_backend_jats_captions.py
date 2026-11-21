# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the placement of figure and table captions in the JATS backend."""

from docling_core.types.doc import DocItemLabel, TextItem

from .test_backend_jats import convert_jats_body


def test_jats_captions_stay_in_the_section_of_their_figure_and_table():
    doc = convert_jats_body(
        """
    <sec><title>One</title>
      <fig id="f1"><label>Figure 1.</label><caption><p>A figure.</p></caption></fig>
      <table-wrap id="t1"><label>Table 1.</label><caption><p>A table.</p></caption>
        <table><tbody><tr><td>a</td><td>b</td></tr></tbody></table>
      </table-wrap>
    </sec>
    <sec><title>Two</title><p>Para two.</p></sec>
"""
    )

    picture, table = doc.pictures[0], doc.tables[0]
    section = picture.parent.resolve(doc)
    assert isinstance(section, TextItem)
    assert section.text == "One"
    assert table.parent == picture.parent

    fig_caption = picture.captions[0].resolve(doc)
    table_caption = table.captions[0].resolve(doc)
    assert fig_caption.text == "Figure 1. A figure."
    assert table_caption.text == "Table 1. A table."
    assert [ref.cref for ref in section.children] == [
        fig_caption.self_ref,
        picture.self_ref,
        table_caption.self_ref,
        table.self_ref,
    ]
    assert all(
        ref.resolve(doc).label != DocItemLabel.CAPTION for ref in doc.body.children
    )
