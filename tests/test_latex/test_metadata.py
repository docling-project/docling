# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Rich LaTeX front matter and its document groups."""

from io import BytesIO

from docling_core.transforms.serializer.latex import LaTeXParams
from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    GroupItem,
)

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

from .latex_serializer import SourceLaTeXDocSerializer, SourceLaTeXInlineSerializer


def _convert(source: bytes) -> DoclingDocument:
    stream = BytesIO(source)
    input_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="metadata.tex",
    )
    return LatexDocumentBackend(in_doc=input_doc, path_or_stream=stream).convert()


def _parent_group_name(doc: DoclingDocument, item) -> str | None:
    if item.parent is None:
        return None
    parent = item.parent.resolve(doc)
    if isinstance(parent, GroupItem) and parent.name in {
        "metadata",
        "authors",
        "affiliations",
    }:
        return parent.name
    return None


def test_latex_front_matter_keeps_roles_and_inline_math():
    source = rb"""\documentclass{article}
\begin{document}
\author{C. Bal\'{a}zs$^{1}$ \and E.~L.~Berger$^{2}$}
\thanks{Current address: Institute $A_2$}
\email{berger@example.org}
\affiliation{$^{1}$Argonne Laboratory \\ $^{2}$Michigan State University}
\altaffiliation[Present address:]{Faculty $B_3$}
\author{\parbox{11cm}{Third Author $^3$}}
\end{document}"""
    doc = _convert(source)

    tagged = [item for item in doc.texts if _parent_group_name(doc, item) is not None]
    assert [_parent_group_name(doc, item) for item in tagged] == [
        "authors",
        "authors",
        "metadata",
        "metadata",
        "affiliations",
        "affiliations",
        "authors",
    ]
    assert tagged[2].label == DocItemLabel.FOOTNOTE
    assert tagged[5].label == DocItemLabel.FOOTNOTE
    assert "11cm" not in doc.export_to_markdown()

    restored = DoclingDocument.model_validate_json(doc.model_dump_json())
    assert [
        _parent_group_name(restored, item)
        for item in restored.texts
        if _parent_group_name(restored, item) is not None
    ] == [_parent_group_name(doc, item) for item in tagged]
    dclg = doc.export_to_doclang()
    assert "Present address:" in dclg
    assert "<latex__classification>" not in dclg
    assert "<formula>^{1}</formula>" in dclg
    assert "<formula>^{2}</formula>" in dclg
    assert "<formula>^3</formula>" in dclg

    parsed = (
        SourceLaTeXDocSerializer(
            doc=doc,
            params=LaTeXParams(
                layers={ContentLayer.BODY, ContentLayer.FURNITURE},
                document_class="",
                packages=[],
                include_annotations=False,
                enable_chart_tables=False,
            ),
            inline_serializer=SourceLaTeXInlineSerializer(),
        )
        .serialize()
        .text
    )
    assert r"C. Balázs$^{1}$" in parsed
    assert r"E. L. Berger$^{2}$" in parsed
    assert r"Third Author $^3$" in parsed
    assert r"\$" not in parsed
    assert "11cm" not in parsed


def test_latex_source_roles_do_not_appear_in_default_markdown():
    doc = _convert(
        rb"\documentclass{article}\title{A title}\author{Alice}\begin{document}Body.\end{document}"
    )

    assert doc.export_to_markdown() == "# A title\n\nAlice\n\nBody."
    metadata = next(group for group in doc.groups if group.name == "metadata")
    authors = next(group for group in doc.groups if group.name == "authors")
    assert authors.parent.cref == metadata.self_ref
    assert all(group.meta is None for group in doc.groups)


def test_latex_preamble_author_keeps_group_and_formula():
    doc = _convert(
        rb"\documentclass{article}\author{Jane $^1$}\begin{document}Body.\end{document}"
    )
    authors = [item for item in doc.texts if _parent_group_name(doc, item) == "authors"]
    assert len(authors) == 1
    assert authors[0].children
    assert "<formula>^1</formula>" in doc.export_to_doclang()


def test_latex_author_inst_markers_are_inline_formulas():
    source = rb"""\documentclass{article}
\begin{document}
\author{Ileana Streinu\inst{1}\thanks{Funded by NSF} \and Louis Theran\inst{2,3}}
\institute{Smith College \and UMass Amherst}
\end{document}"""
    doc = _convert(source)

    authors = [item for item in doc.texts if _parent_group_name(doc, item) == "authors"]
    assert len(authors) == 2
    assert all(item.children for item in authors)
    dclg = doc.export_to_doclang()
    assert "<formula>^{1}</formula>" in dclg
    assert "<formula>^{2,3}</formula>" in dclg
    assert "Ileana Streinu1" not in dclg
    assert "Louis Theran2,3" not in dclg

    parsed = (
        SourceLaTeXDocSerializer(
            doc=doc,
            params=LaTeXParams(
                layers={ContentLayer.BODY, ContentLayer.FURNITURE},
                document_class="",
                packages=[],
                include_annotations=False,
                enable_chart_tables=False,
            ),
            inline_serializer=SourceLaTeXInlineSerializer(),
        )
        .serialize()
        .text
    )
    assert r"Ileana Streinu$^{1}$" in parsed
    assert r"Louis Theran$^{2,3}$" in parsed
    assert "Funded by NSF" in parsed
