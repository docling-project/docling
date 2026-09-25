# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Backends for Apple Pages (``.pages``) and Keynote (``.key``) documents.

Either file is a ZIP container, but what is inside changed completely with the
2013 releases:

* **Pages 5 / Keynote 6 and later (2013 onwards)** store the document as
  ``Index/*.iwa`` — Snappy-framed protobuf whose schemas Apple has never
  published. This is what essentially every iWork document in circulation looks
  like. Keynote 2018 and later flatten the package into a subdirectory and zip
  that index a second time, into an ``Index.zip``.
* **iWork '09 and earlier** stored it as plain XML — ``index.xml`` for Pages and
  ``index.apxl`` for Keynote, either of them optionally gzipped — alongside a
  ``QuickLook/Preview.pdf`` render that Apple stopped writing after that release.

Every generation is read into the same model, so the backends are declarative:
they build a :class:`~docling_core.types.doc.DoclingDocument` directly rather
than rendering pages and running layout analysis over them.
"""

import logging
import mimetypes
import zipfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlparse

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    ImageRef,
    NodeItem,
    PictureClassificationLabel,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
    TabularChartMetaField,
)
from docling_core.types.doc.items.group import ListGroup
from docling_core.types.doc.items.text import TextItem
from PIL import Image
from pydantic import AnyUrl, ValidationError
from typing_extensions import override

from docling.backend.abstract_backend import (
    DeclarativeDocumentBackend,
    PaginatedDocumentBackend,
)
from docling.backend.docx.drawingml.utils import get_docx_to_pdf_converter
from docling.backend.iwork import (
    chart_image,
    keynote_iwa,
    keynote_xml,
    pages_iwa,
    pages_xml,
)
from docling.backend.iwork.content import (
    Block,
    Chart,
    ChartKind,
    Comment,
    Content,
    Geometry,
    ListLabel,
    Paragraph,
    Picture,
    Run,
)
from docling.backend.iwork.iwa import is_encrypted
from docling.backend.iwork.keynote_content import Presentation, Slide
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

_T = TypeVar("_T")

_PAGES_MIMETYPE = "application/vnd.apple.pages"

_KEYNOTE_MIMETYPE = "application/vnd.apple.keynote"

_PAGES_KIND = "Pages"

_KEYNOTE_KIND = "Keynote"

_MODERN_INDEX_PREFIX = "Index/"

_NESTED_INDEX_MEMBER = "Index.zip"
"""The index of a package Keynote 2018 and later flattened into one file.

The ``Index/`` directory is zipped a second time and put in a subdirectory of
the container, while the ``Data/`` members stay unzipped beside it — so the two
halves of such a document are read out of two different archives.
"""

_LEGACY_INDEX_MEMBERS = ("index.xml", "index.xml.gz")

_KEYNOTE_LEGACY_INDEX_MEMBERS = ("index.apxl", "index.apxl.gz")

_CHART_RENDER_HINT = (
    "LibreOffice is required to render Keynote charts as images "
    "(render_chart_images=True): each chart is rebuilt as an Office chart for "
    "LibreOffice to draw. Install LibreOffice and make sure `soffice` is on PATH. "
    "Charts still keep their classification and data."
)

_ChartRenderer = Callable[[Chart, Geometry | None], ImageRef | None]
"""Draws a chart, given where it sits, or returns None when it cannot."""

_CHART_LABELS = {
    ChartKind.COLUMN: PictureClassificationLabel.BAR_CHART,
    ChartKind.BAR: PictureClassificationLabel.BAR_CHART,
    ChartKind.LINE: PictureClassificationLabel.LINE_CHART,
    ChartKind.PIE: PictureClassificationLabel.PIE_CHART,
    ChartKind.DONUT: PictureClassificationLabel.PIE_CHART,
    ChartKind.SCATTER: PictureClassificationLabel.SCATTER_CHART,
}
"""How a chart is classified, by its kind; anything else is ``OTHER_CHART``.

The same families the PowerPoint and Excel backends classify into, so a chart
reads the same whichever of them it came from: column and bar charts, stacked
or not, are bar charts, a donut is a pie, and area, bubble, radar and mixed
charts are other charts.
"""


def _open_container(
    path_or_stream: BytesIO | Path,
    read: Callable[[zipfile.ZipFile], _T],
    kind: str,
    document_hash: str,
) -> _T:
    """Open an iWork container and read it, reporting why it could not be.

    Every way a container can defeat the readers surfaces here, so that both
    backends fail the same way and a caller gets a message about the file rather
    than a traceback out of zipfile, zlib or the protobuf walk.

    Args:
        path_or_stream: The document to open.
        read: Reads the open container into whatever the backend models it as.
        kind: What the app calls its documents, for error messages.
        document_hash: The document's hash, for error messages.

    Returns:
        Whatever ``read`` returned.

    Raises:
        DocumentLoadError: If the container cannot be opened or read.
    """
    try:
        with zipfile.ZipFile(path_or_stream) as archive:
            return read(archive)
    except DocumentLoadError:
        raise
    except RecursionError as exc:
        # RecursionError subclasses RuntimeError, so it must be caught first;
        # otherwise deeply nested XML would be reported as an encryption
        # problem, hiding a real robustness failure behind benign advice.
        raise DocumentLoadError(
            f"{kind} document with hash {document_hash} is nested too deeply to parse."
        ) from exc
    except (NotImplementedError, RuntimeError) as exc:
        # Encryption is normally detected up front from the member table.
        # Anything reaching here is an unreadable member for some other
        # reason (an unknown compression method, a missing codec module), so
        # the message stays about the container rather than about passwords.
        raise DocumentLoadError(
            f"Could not read {kind} document with hash {document_hash}: "
            f"the archive contains a member Docling cannot decompress ({exc})."
        ) from exc
    except (zipfile.BadZipFile, OSError) as exc:
        raise DocumentLoadError(
            f"Could not open {kind} document with hash {document_hash}: "
            "the file is not a readable ZIP container."
        ) from exc


def _is_nested_index(name: str) -> bool:
    """Report whether a member is the index of a flattened package.

    Keynote puts it at the root of the one directory it flattens the package
    into, so anything deeper is something else that happens to be called
    ``Index.zip`` — a zipped index the author dropped into the deck, say.

    Args:
        name: An archive member's name.

    Returns:
        Whether it is where the index of a flattened package would be.
    """
    return name.endswith(_NESTED_INDEX_MEMBER) and name.count("/") <= 1


def _readable_members(
    archive: zipfile.ZipFile,
    options: IWorkBackendOptions,
    kind: str,
    document_hash: str,
) -> list[zipfile.ZipInfo]:
    """Return a container's members, refusing one this is not willing to read.

    Args:
        archive: The open container.
        options: The limits to hold it to.
        kind: What the app calls its documents, for error messages.
        document_hash: The document's hash, for error messages.

    Returns:
        The container's members.

    Raises:
        DocumentLoadError: If the container has too many members, expands too
            far, or is password-protected.
    """
    infos = archive.infolist()
    if len(infos) > options.max_member_count:
        raise DocumentLoadError(
            f"{kind} archive has {len(infos)} members, exceeding the "
            f"max_member_count limit of {options.max_member_count}."
        )
    total_bytes = sum(info.file_size for info in infos)
    if total_bytes > options.max_total_bytes:
        raise DocumentLoadError(
            f"{kind} archive expands to {total_bytes} bytes, exceeding the "
            f"max_total_bytes limit of {options.max_total_bytes}."
        )

    if any(is_encrypted(info) for info in infos):
        raise DocumentLoadError(
            f"{kind} document with hash {document_hash} is "
            "password-protected; Docling cannot read encrypted iWork "
            f"documents. Remove the password in {kind} and save again."
        )

    return infos


class IWorkPagesDocumentBackend(DeclarativeDocumentBackend):
    """Extract text from Apple Pages documents of either generation.

    Known limitations:
        * Only text cells are read from a table, in either of the two storage
          layouts Pages has used. A cell holding a number, a date or a formula
          result is left empty rather than guessed at.
        * A picture is placed where the document anchors it, but its caption,
          its cropping and its accessibility description are not read.
        * Bold, italic, underline, strikethrough, superscript, subscript and
          hyperlinks are recovered; other character properties, such as colour
          or capitalisation, have no equivalent here.
        * A list item whose runs differ in formatting keeps its text but loses
          the formatting, since a list item carries a single one.
        * Text boxes are read from Pages 5+ documents, where they are floating
          drawables owned by the document. An iWork '09 document keeps them in
          the body flow, so they already appear there.
        * Headers, footers and footnotes are recovered into the furniture
          content layer and comments into the notes layer, so all of them stay
          out of the reading order by default.
        * A comment records its author but not the date it was written.
        * Password-protected documents cannot be read.
        * ``.pages`` bundles saved as a *directory* package rather than a single
          file are not recognised; the converter cannot address a directory as an
          input document.
    """

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: IWorkBackendOptions | None = None,
    ):
        if options is None:
            options = IWorkBackendOptions()
        super().__init__(in_doc, path_or_stream, options)
        self.options: IWorkBackendOptions = options

        self._content = Content(blocks=[])
        self._valid = False

        self._content = _open_container(
            path_or_stream, self._read_document, _PAGES_KIND, self.document_hash
        )
        self._valid = True

    def _read_document(self, archive: zipfile.ZipFile) -> Content:
        """Dispatch to the reader for whichever generation wrote the container."""
        infos = _readable_members(
            archive, self.options, _PAGES_KIND, self.document_hash
        )

        names = {info.filename for info in infos}
        if any(name.startswith(_MODERN_INDEX_PREFIX) for name in names):
            return pages_iwa.read_content(
                archive, infos, self.options.max_file_bytes, self.document_hash
            )

        legacy = next((n for n in _LEGACY_INDEX_MEMBERS if n in names), None)
        if legacy is not None:
            return pages_xml.read_content(
                archive, legacy, self.options.max_total_bytes, self.document_hash
            )

        raise DocumentLoadError(
            f"Document with hash {self.document_hash} is a ZIP archive but does "
            "not look like a Pages document: it has neither an Index/ directory "
            "nor an index.xml."
        )

    @override
    def is_valid(self) -> bool:
        return self._valid

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.IWORK_PAGES}

    @override
    def convert(self) -> DoclingDocument:
        if not self.is_valid():
            raise RuntimeError(
                f"Cannot convert Pages document with hash {self.document_hash} "
                "because the backend failed to init."
            )

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype=_PAGES_MIMETYPE,
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        lists = _ListStack(doc)
        annotated: dict[str, TextItem] = {}
        for block in self._content.blocks:
            item = _add_block(doc, block, lists)
            if isinstance(block, Paragraph) and item is not None:
                for anchor in block.anchors:
                    annotated.setdefault(anchor, item)

        _add_furniture(doc, self._content)
        _add_comments(doc, self._content.comments, annotated)
        return doc


class IWorkKeynoteDocumentBackend(DeclarativeDocumentBackend, PaginatedDocumentBackend):
    """Extract slides from Apple Keynote presentations of any generation.

    A slide becomes a chapter group holding what was placed on it, in the order
    it is read rather than the order Keynote stacked it, and a page of the
    slide's own size, which is the shape the PowerPoint and ODP backends give a
    presentation. Presenter notes and comments go into the notes content layer
    under the slide they belong to, so they stay out of the reading order by
    default.

    Known limitations:
        * Only text cells are read from a table, in either of the two storage
          layouts Keynote has used. A cell holding a number, a date or a formula
          result is left empty rather than guessed at.
        * A picture is placed where the slide anchors it, but its caption, its
          cropping and its accessibility description are not read.
        * A chart becomes a picture classified by its kind, carrying the data it
          plots as a table and captioned with its title, as the PowerPoint
          backend gives one. Keynote keeps no picture of a chart, so the
          picture is empty unless ``render_chart_images`` redraws one from that
          data, without the original's colours and fonts. A value that is a
          date or a duration rather than a number is left empty, and an iWork
          '09 chart is not read.
        * What a master slide draws is left to the master: it belongs to every
          slide using it rather than to any one of them, so it is not repeated.
          A slide that shows nothing of its own therefore yields an empty group.
        * Bold, italic, underline, strikethrough, superscript, subscript and
          hyperlinks are recovered; other character properties, such as colour
          or capitalisation, have no equivalent here.
        * A slide title is recovered from the placeholder Keynote reserves for
          it. Text in any other shape is body text, whatever the theme calls the
          style it carries, since those names are localised.
        * A comment records its text but not its author or the date it was
          written, and is attached to the slide rather than to a stretch of it:
          Keynote draws one as a note stuck to the slide rather than as a
          highlight over words.
        * Builds, transitions and the order they animate a slide's contents in
          are not read, so a slide's blocks are ordered by where they sit.
        * Password-protected presentations cannot be read.
        * ``.key`` bundles saved as a *directory* package rather than a single
          file are not recognised; the converter cannot address a directory as an
          input document.
    """

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: IWorkBackendOptions | None = None,
    ):
        if options is None:
            options = IWorkBackendOptions()
        super().__init__(in_doc, path_or_stream, options)
        self.options: IWorkBackendOptions = options

        self._presentation = _open_container(
            path_or_stream, self._read_document, _KEYNOTE_KIND, self.document_hash
        )
        self._valid = True

    def _read_document(self, archive: zipfile.ZipFile) -> Presentation:
        """Dispatch to the reader for whichever generation wrote the container."""
        infos = _readable_members(
            archive, self.options, _KEYNOTE_KIND, self.document_hash
        )

        names = {info.filename for info in infos}
        if any(name.startswith(_MODERN_INDEX_PREFIX) for name in names):
            return keynote_iwa.read_content(
                archive,
                infos,
                archive,
                "",
                self.options.max_file_bytes,
                self.document_hash,
            )

        nested = next(
            (name for name in sorted(names) if _is_nested_index(name)),
            None,
        )
        if nested is not None:
            return self._read_nested(archive, nested)

        legacy = next(
            (name for name in _KEYNOTE_LEGACY_INDEX_MEMBERS if name in names), None
        )
        if legacy is not None:
            return keynote_xml.read_content(
                archive, legacy, self.options.max_total_bytes, self.document_hash
            )

        raise DocumentLoadError(
            f"Document with hash {self.document_hash} is a ZIP archive but does "
            "not look like a Keynote document: it has neither an Index/ "
            "directory nor an Index.zip nor an index.apxl."
        )

    def _read_nested(self, archive: zipfile.ZipFile, member: str) -> Presentation:
        """Read a presentation whose index was zipped a second time.

        The inner archive holds the object graph and the outer one the image
        data, under the same prefix the inner archive was found at, so both are
        passed to the reader.

        Everything the index holds is inside the inner archive, so it is held to
        the same limits as a container whose index was not nested — the stored
        size of the ``Index.zip`` member says nothing about what is in it, and a
        2 KiB one can expand to hundreds of megabytes.

        Args:
            archive: The open ``.key`` container.
            member: The name of its ``Index.zip`` member.

        Returns:
            Everything the presentation holds.

        Raises:
            DocumentLoadError: If either archive is larger or holds more
                members than this is willing to read, or is password-protected.
        """
        size = archive.getinfo(member).file_size
        if size > self.options.max_file_bytes:
            raise DocumentLoadError(
                f"Keynote archive member {member} is {size} bytes, exceeding "
                f"the max_file_bytes limit of {self.options.max_file_bytes}."
            )

        with zipfile.ZipFile(BytesIO(archive.read(member))) as index:
            infos = _readable_members(
                index, self.options, _KEYNOTE_KIND, self.document_hash
            )
            return keynote_iwa.read_content(
                index,
                infos,
                archive,
                member[: -len(_NESTED_INDEX_MEMBER)],
                self.options.max_file_bytes,
                self.document_hash,
            )

    @override
    def is_valid(self) -> bool:
        return self._valid

    @override
    def page_count(self) -> int:
        return len(self._presentation.slides) if self.is_valid() else 0

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return True

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.IWORK_KEYNOTE}

    @override
    def convert(self) -> DoclingDocument:
        if not self.is_valid():
            raise RuntimeError(
                f"Cannot convert Keynote document with hash {self.document_hash} "
                "because the backend failed to init."
            )

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype=_KEYNOTE_MIMETYPE,
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        size = Size(width=self._presentation.width, height=self._presentation.height)

        render_chart = self._chart_renderer()
        for index, slide in enumerate(self._presentation.slides):
            doc.add_page(page_no=index + 1, size=size)
            group = doc.add_group(name=f"slide-{index}", label=GroupLabel.CHAPTER)
            _add_slide(doc, slide, group, index + 1, render_chart)

        return doc

    def _chart_renderer(self) -> _ChartRenderer | None:
        """Return what draws the presentation's charts, if the caller asked for it.

        Returns:
            A renderer, or None when rendering is off or cannot run here, which
            is reported once rather than once per chart.
        """
        if not self.options.render_chart_images:
            return None
        converter = get_docx_to_pdf_converter()
        if converter is None:
            _log.warning(_CHART_RENDER_HINT)
            return None

        def render(chart: Chart, geometry: Geometry | None) -> ImageRef | None:
            image = chart_image.render_chart(chart, geometry, converter)
            return ImageRef.from_pil(image=image, dpi=72) if image is not None else None

        return render


class _ListStack:
    """The list groups open while consecutive list items keep arriving.

    Pages records a nesting depth per paragraph rather than opening and closing
    lists, so the groups a :class:`DoclingDocument` needs are inferred here: a
    deeper item opens groups down to its depth, a shallower one closes back to
    it, and any other paragraph ends the list entirely.
    """

    def __init__(self, doc: DoclingDocument, parent: NodeItem | None = None) -> None:
        self._doc = doc
        self._parent = parent
        self._groups: list[ListGroup] = []

    def close(self) -> None:
        """End the list, so the next item starts a new one."""
        self._groups.clear()

    def group_for(self, depth: int) -> ListGroup:
        """Return the group a list item at ``depth`` belongs in, opening it if needed.

        Args:
            depth: The item's nesting depth, counted from zero.

        Returns:
            The innermost open group.
        """
        del self._groups[depth + 1 :]
        while len(self._groups) <= depth:
            self._groups.append(
                self._doc.add_list_group(
                    name="list",
                    parent=self._groups[-1] if self._groups else self._parent,
                )
            )
        return self._groups[depth]


def _add_slide(
    doc: DoclingDocument,
    slide: Slide,
    group: NodeItem,
    page_no: int,
    render_chart: _ChartRenderer | None = None,
) -> None:
    """Add one slide's contents, its presenter notes and its comments.

    Args:
        doc: The document being built.
        slide: The slide to add.
        group: The group standing for the slide.
        page_no: The page the slide is, counted from one.
        render_chart: Draws the slide's charts, or None to leave them undrawn.
    """
    lists = _ListStack(doc, group)
    for placed in slide.blocks:
        image = (
            render_chart(placed.block, placed.geometry)
            if render_chart is not None and isinstance(placed.block, Chart)
            else None
        )
        _add_block(
            doc,
            placed.block,
            lists,
            parent=group,
            prov=_slide_prov(placed.geometry, page_no, _block_text(placed.block)),
            image=image,
        )

    for note in slide.notes:
        _add_runs(
            doc,
            note,
            DocItemLabel.TEXT,
            content_layer=ContentLayer.NOTES,
            parent=group,
            prov=_slide_prov(None, page_no, note.text),
        )

    for comment in slide.comments:
        doc.add_comment(text=comment.text, parent=group)


def _block_text(block: Block) -> str:
    """The text a block carries, which is none unless it is a paragraph."""
    return block.text if isinstance(block, Paragraph) else ""


def _slide_prov(geometry: Geometry | None, page_no: int, text: str) -> ProvenanceItem:
    """Place an item on the slide it was read from.

    Args:
        geometry: Where the drawable holding it sits, if Keynote positioned one.
        page_no: The slide's page number, counted from one.
        text: The item's text, whose length is the span recorded.

    Returns:
        The provenance. A drawable Keynote did not position, and a presenter
        note, which is not drawn on the slide at all, get an empty box rather
        than one covering the whole slide: the page is what makes them
        addressable, and a box that was never measured would not.
    """
    if geometry is None:
        bbox = BoundingBox(l=0, t=0, r=0, b=0, coord_origin=CoordOrigin.TOPLEFT)
    else:
        bbox = BoundingBox(
            l=geometry.left,
            t=geometry.top,
            r=geometry.left + geometry.width,
            b=geometry.top + geometry.height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
    return ProvenanceItem(page_no=page_no, charspan=(0, len(text)), bbox=bbox)


def _add_comments(
    doc: DoclingDocument, comments: list[Comment], annotated: dict[str, TextItem]
) -> None:
    """Add the document's comments, linked to the text they annotate.

    Comments go into the notes content layer, where the Word backend puts them
    too, and each is attached to the item holding the text it was written about
    whenever that text was recovered.

    Args:
        doc: The document being built.
        comments: The comments read from the Pages document.
        annotated: The item each comment anchor was recovered into.
    """
    for comment in comments:
        target = annotated.get(comment.anchor)
        doc.add_comment(text=comment.text, targets=[target] if target else None)


def _add_furniture(doc: DoclingDocument, content: Content) -> None:
    """Add the document's headers, footers and footnotes.

    They go into the furniture layer, where the Word backend puts a header and a
    footer too, so they are available to callers that ask for it but stay out of
    the reading order by default.

    Args:
        doc: The document being built.
        content: The content read from the Pages document.
    """
    for paragraphs, label in (
        (content.headers, DocItemLabel.PAGE_HEADER),
        (content.footers, DocItemLabel.PAGE_FOOTER),
        (content.footnotes, DocItemLabel.FOOTNOTE),
    ):
        for paragraph in paragraphs:
            _add_runs(doc, paragraph, label, ContentLayer.FURNITURE)


def _add_block(
    doc: DoclingDocument,
    block: Block,
    lists: _ListStack,
    parent: NodeItem | None = None,
    prov: ProvenanceItem | None = None,
    image: ImageRef | None = None,
) -> TextItem | None:
    """Add one block of content, in the order the document lays it out.

    Args:
        doc: The document being built.
        block: The block to add.
        lists: The list groups currently open.
        parent: The node to add it under, or None for the document root.
        prov: Where the block came from, for the backends that know.
        image: A picture drawn of a chart, which the document holds none of.

    Returns:
        The item a paragraph became, so a comment can be attached to it, or None
        for anything a comment cannot annotate.
    """
    if isinstance(block, Paragraph):
        return _add_paragraph(doc, block, lists, parent, prov)

    # A table, a picture or a chart ends any list it follows, like body text.
    lists.close()
    if isinstance(block, Picture):
        _add_picture(doc, block, parent, prov)
    elif isinstance(block, Chart):
        _add_chart(doc, block, parent, prov, image)
    else:
        doc.add_table(data=block, parent=parent, prov=prov)
    return None


def _add_picture(
    doc: DoclingDocument,
    picture: Picture,
    parent: NodeItem | None = None,
    prov: ProvenanceItem | None = None,
) -> None:
    """Add one picture, embedding its image when the bytes can be decoded.

    Args:
        doc: The document being built.
        picture: The picture to add.
        parent: The node to add it under, or None for the document root.
        prov: Where the picture came from, for the backends that know.
    """
    image: ImageRef | None = None
    if picture.data is not None:
        try:
            with Image.open(BytesIO(picture.data)) as opened:
                image = ImageRef.from_pil(image=opened.convert("RGB"), dpi=72)
        except (OSError, ValueError) as exc:
            # Pages stores whatever the author placed, including formats Pillow
            # has no decoder for. The picture still belongs in the flow.
            _log.debug("Could not decode iWork image %s: %s", picture.name, exc)

    doc.add_picture(image=image, parent=parent, prov=prov)


def _add_chart(
    doc: DoclingDocument,
    chart: Chart,
    parent: NodeItem | None = None,
    prov: ProvenanceItem | None = None,
    image: ImageRef | None = None,
) -> None:
    """Add one chart as a picture classified by its kind and carrying its data.

    This is the shape the PowerPoint backend gives a chart: a picture whose meta
    holds the chart's classification and its data as a table, captioned with
    the chart's title when the chart shows one.

    Args:
        doc: The document being built.
        chart: The chart to add.
        parent: The node to add it under, or None for the document root.
        prov: Where the chart came from, for the backends that know.
        image: A picture drawn of the chart, or None to leave it undrawn.
    """
    caption = None
    if chart.title:
        caption = doc.add_text(
            label=DocItemLabel.CAPTION,
            text=chart.title,
            parent=parent,
            prov=(
                prov.model_copy(update={"charspan": (0, len(chart.title))})
                if prov is not None
                else None
            ),
        )

    picture = doc.add_picture(image=image, caption=caption, parent=parent, prov=prov)
    label = _CHART_LABELS.get(chart.kind, PictureClassificationLabel.OTHER_CHART)
    table = _chart_table(chart)
    picture.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[PictureClassificationPrediction(class_name=label)]
        ),
        tabular_chart=(
            TabularChartMetaField(chart_data=table) if table is not None else None
        ),
    )


def _chart_table(chart: Chart) -> TableData | None:
    """Lay a chart's data out as a table, categories down and series across.

    It is the layout the PowerPoint and Excel backends give a chart's data, so a
    consumer reads every chart's table the same way::

        | <blank> | <series 0 name> | <series 1 name> | ...
        | cat_0   | val_0,0         | val_1,0         | ...
        | cat_1   | val_0,1         | val_1,1         | ...

    Args:
        chart: The chart whose data to lay out.

    Returns:
        The table, or None when the chart holds no data.
    """
    rows = max([len(chart.categories)] + [len(s.values) for s in chart.series])
    if not chart.series or rows == 0:
        return None

    texts = [["", *(series.name for series in chart.series)]]
    for row in range(rows):
        category = chart.categories[row] if row < len(chart.categories) else ""
        values = (s.values[row] if row < len(s.values) else None for s in chart.series)
        texts.append([category, *map(_chart_value, values)])

    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=col,
            end_col_offset_idx=col + 1,
            column_header=row == 0,
            row_header=row > 0 and col == 0,
        )
        for row, line in enumerate(texts)
        for col, text in enumerate(line)
    ]
    return TableData(
        num_rows=rows + 1, num_cols=len(chart.series) + 1, table_cells=cells
    )


def _chart_value(value: float | None) -> str:
    """Write a chart value the way the chart's data editor shows it.

    A whole number loses the ``.0`` a float would print with, which is what the
    PowerPoint backend does too, so ``120`` reads as it was typed.
    """
    if value is None:
        return ""
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return str(value)


def _add_paragraph(
    doc: DoclingDocument,
    paragraph: Paragraph,
    lists: _ListStack,
    parent: NodeItem | None = None,
    prov: ProvenanceItem | None = None,
) -> TextItem | None:
    """Add one paragraph, as a heading, a list item or body text.

    Args:
        doc: The document being built.
        paragraph: The paragraph to add.
        lists: The list groups currently open.
        parent: The node to add it under, or None for the document root.
        prov: Where the paragraph came from, for the backends that know.

    Returns:
        The item the paragraph became, or its first when its runs differ.
    """
    if paragraph.list_label is None:
        lists.close()
    else:
        return _add_list_item(doc, paragraph, paragraph.list_label, lists, prov)

    if paragraph.label == DocItemLabel.TITLE:
        return doc.add_title(text=paragraph.text, parent=parent, prov=prov)
    if paragraph.label == DocItemLabel.SECTION_HEADER:
        return doc.add_heading(
            text=paragraph.text,
            level=paragraph.level or 1,
            parent=parent,
            prov=prov,
        )

    return _add_runs(doc, paragraph, paragraph.label, parent=parent, prov=prov)


def _add_runs(
    doc: DoclingDocument,
    paragraph: Paragraph,
    label: DocItemLabel,
    content_layer: ContentLayer | None = None,
    parent: NodeItem | None = None,
    prov: ProvenanceItem | None = None,
) -> TextItem:
    """Add a paragraph's runs, keeping the formatting attached to each one.

    ``TextItem`` carries a single ``Formatting`` and a single hyperlink, so a
    paragraph whose runs differ in either has to become an inline group of items
    — the same shape the Word and HTML backends produce for mixed runs.

    Args:
        doc: The document being built.
        paragraph: The paragraph to add.
        label: The label to give the item, or every item of the group.
        content_layer: The layer to add to, or None for the document's default.
        parent: The node to add it under, or None for the document root.
        prov: Where the paragraph came from, for the backends that know.

    Returns:
        The item added, or the first of the group. A comment annotates a stretch
        of the paragraph, and the first item is the one that always exists,
        whichever run that stretch began in.
    """
    runs = [run for run in paragraph.runs if run.text]
    if _uniform(runs):
        first = runs[0] if runs else Run("", None)
        return doc.add_text(
            label=label,
            text=paragraph.text,
            formatting=first.formatting,
            hyperlink=_hyperlink(first.hyperlink),
            content_layer=content_layer,
            parent=parent,
            prov=prov,
        )

    group = doc.add_inline_group(content_layer=content_layer, parent=parent)
    items = [
        doc.add_text(
            label=label,
            text=run.text,
            formatting=run.formatting,
            hyperlink=_hyperlink(run.hyperlink),
            parent=group,
            content_layer=content_layer,
            prov=prov,
        )
        for run in runs
    ]
    return items[0]


def _add_list_item(
    doc: DoclingDocument,
    paragraph: Paragraph,
    label: ListLabel,
    lists: _ListStack,
    prov: ProvenanceItem | None = None,
) -> TextItem:
    """Add one list item under the group its nesting depth belongs to."""
    group = lists.group_for(label.depth)
    runs = [run for run in paragraph.runs if run.text]
    uniform = runs[0] if runs and _uniform(runs) else Run("", None)
    return doc.add_list_item(
        text=paragraph.text,
        enumerated=label.enumerated,
        marker=label.marker,
        parent=group,
        formatting=uniform.formatting,
        hyperlink=_hyperlink(uniform.hyperlink),
        prov=prov,
    )


def _uniform(runs: list[Run]) -> bool:
    """Report whether every run shares one formatting and one link.

    ``Formatting`` is a model rather than a hashable value, so the runs are
    compared against the first rather than deduplicated through a set.
    """
    if len(runs) <= 1:
        return True
    first = runs[0]
    return all(
        run.formatting == first.formatting and run.hyperlink == first.hyperlink
        for run in runs
    )


def _hyperlink(address: str | None) -> AnyUrl | Path | None:
    """Resolve a link's address to a URL or a local path.

    A Pages document can link to a file next to it as well as to a URL, and an
    address Pydantic will not accept is dropped rather than allowed to fail the
    whole conversion.

    Args:
        address: The address the document recorded, if any.

    Returns:
        The address as a URL or a path, or None when there is none to use.
    """
    if not address:
        return None
    if not urlparse(address).scheme:
        return Path(address)
    try:
        return AnyUrl(address)
    except ValidationError:
        _log.debug("Skipping malformed Pages hyperlink address: %r", address)
        return None
