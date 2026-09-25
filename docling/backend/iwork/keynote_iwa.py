# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reader for the object graph of a Keynote 6+ (2013 onwards) presentation.

The container is a set of ``Index/*.iwa`` archives, the same ones a Pages
document is written into, so everything below a slide — text storages, tables,
images — is read by :mod:`docling.backend.iwork.archives`. What ``KN``,
Keynote's own namespace, adds on top is a show holding a tree of slide nodes,
and a slide holding placeholders: a title, a body, a slide number, and whatever
else was dropped onto it.

A placeholder is what makes a slide title recoverable. Its *style* is no help:
the style names are localised, and one of the test fixtures names them in
Indonesian, so the title is taken from the placeholder Keynote reserves for it
rather than from a style called "Title".
"""

import zipfile

from docling_core.types.doc import DocItemLabel

from docling.backend.iwork.archives import (
    SHAPE_STORAGE_FIELD,
    TSWP_STORAGE_ARCHIVE,
    IWAReader,
    drawable_geometry,
    iwa_reference_field,
    iwa_reference_list,
    read_objects,
    read_point,
    safe_fields,
)
from docling.backend.iwork.charts import TSCH_CHART_DRAWABLE, iwa_chart
from docling.backend.iwork.content import Block, Comment, Geometry, Paragraph
from docling.backend.iwork.iwa import IWAObject
from docling.backend.iwork.keynote_content import (
    DEFAULT_SLIDE_HEIGHT,
    DEFAULT_SLIDE_WIDTH,
    Placed,
    Presentation,
    Slide,
    reading_order,
)
from docling.exceptions import DocumentLoadError

KEYNOTE_KIND = "Keynote"
"""What Keynote calls its documents, for the error messages of a shared reader."""

KN_DOCUMENT_ARCHIVE = 1
"""Message type of ``KN.DocumentArchive``, the root object of a presentation."""

KN_SHOW_ARCHIVE = 2
"""Message type of ``KN.ShowArchive``, which holds the slides and their size."""

KN_SLIDE_NODE_ARCHIVE = 4
"""Message type of ``KN.SlideNodeArchive``, one entry of the slide tree.

The tree is what lets a slide be grouped under another one in the navigator; a
node carries that structure and points at the slide itself.
"""

KN_SLIDE_ARCHIVE = 5
"""Message type of ``KN.SlideArchive``, one slide."""

KN_PLACEHOLDER_ARCHIVE = 7
"""Message type of ``KN.PlaceholderArchive``, one of a slide's reserved shapes.

It wraps a ``TSWP.ShapeInfoArchive`` rather than subclassing the drawable
directly, so its text is reached through that super rather than from the
placeholder itself.
"""

KN_NOTE_ARCHIVE = 15
"""Message type of ``KN.NoteArchive``, the presenter notes of one slide."""

KN_COMMENT_ARCHIVE = 2014
"""Message type of the shape a Keynote comment is drawn in.

Keynote shows a comment as a sticky note stuck to the slide, so it is a drawable
in the slide's own list, not an annotation over a stretch of text the way Pages
records one. The shape holds a copy of the comment's text for drawing and points
at the ``TSD.CommentStorageArchive`` that holds the comment proper, which is
where the author and any replies are — so the shape is read as a comment and
never as body text, or the comment would be emitted twice.
"""

DOCUMENT_SHOW_FIELD = 2
"""Field of ``KN.DocumentArchive`` referencing its ``KN.ShowArchive``."""

SHOW_SLIDE_TREE_FIELD = 3
"""Field of ``KN.ShowArchive`` holding the slide tree.

The tree is a message rather than a reference, and its own repeated field points
at the slide nodes. The theme keeps a second list of the same shape holding the
master slides, which is why the slides are reached from the show and not by
collecting every ``KN.SlideArchive`` in the container.
"""

SHOW_SLIDE_SIZE_FIELD = 4
"""Field of ``KN.ShowArchive`` holding the slide size, as a ``TSP.Size``."""

SLIDE_TREE_NODES_FIELD = 2
"""Field of the slide tree referencing its nodes, in presentation order."""

SLIDE_NODE_SLIDE_FIELD = 2
"""Field of ``KN.SlideNodeArchive`` referencing the slide it stands for."""

SLIDE_TITLE_FIELD = 5

SLIDE_BODY_FIELD = 6
"""Fields of ``KN.SlideArchive`` referencing its title and body placeholders.

Both are written whether or not the author typed anything into them, so an empty
one is ordinary and yields nothing rather than an empty item.
"""

SLIDE_DRAWABLES_FIELD = 7
"""Field of ``KN.SlideArchive`` referencing everything placed on the slide.

It repeats the placeholders the slide names separately, and sometimes omits one
of them, so the two sources are merged rather than either being trusted alone.
"""

SLIDE_NUMBER_FIELD = 20
"""Field of ``KN.SlideArchive`` referencing its slide-number placeholder.

It appears in the drawables list too, so it is named here in order to be left
out: a slide number is page furniture, and the placeholder holds nothing but the
U+FFFC the number is substituted for anyway.
"""

SLIDE_NOTE_FIELD = 27
"""Field of ``KN.SlideArchive`` referencing its ``KN.NoteArchive``."""

PLACEHOLDER_SHAPE_FIELD = 1
"""Field of ``KN.PlaceholderArchive`` holding its ``TSWP.ShapeInfoArchive``."""

NOTE_TEXT_FIELD = 1
"""Field of ``KN.NoteArchive`` referencing the storage holding the notes."""

COMMENT_STORAGE_FIELD = 2
"""Field of a comment shape referencing its ``TSD.CommentStorageArchive``."""


def read_content(
    index: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    container: zipfile.ZipFile,
    data_prefix: str,
    max_file_bytes: int,
    document_hash: str,
) -> Presentation:
    """Read a Keynote 6+ presentation out of its IWA object graph.

    Args:
        index: The archive holding the ``.iwa`` members, which is the container
            itself unless Keynote nested the index in an ``Index.zip``.
        infos: That archive's members.
        container: The ``.key`` container, which holds the image data either way.
        data_prefix: What ``container`` puts in front of its ``Data/`` members.
        max_file_bytes: The largest member this is willing to decompress.
        document_hash: The document's hash, for error messages.

    Returns:
        Every slide the presentation holds, and the size they are laid out at.

    Raises:
        DocumentLoadError: If a member is too large, or the object graph has no
            document archive or no show.
    """
    objects = read_objects(index, infos, max_file_bytes, KEYNOTE_KIND)

    document = next(
        (o for o in objects.values() if o.message_type == KN_DOCUMENT_ARCHIVE),
        None,
    )
    if document is None:
        raise DocumentLoadError(
            f"Keynote document with hash {document_hash} has no "
            "KN.DocumentArchive; the container may be corrupt or "
            "password-protected."
        )

    target = iwa_reference_field(document.payload, DOCUMENT_SHOW_FIELD)
    show = objects.get(target) if target is not None else None
    if show is None or show.message_type != KN_SHOW_ARCHIVE:
        raise DocumentLoadError(
            f"Keynote document with hash {document_hash} does not reference a "
            "KN.ShowArchive, so it has no slides to read."
        )

    reader = KeynoteReader(container, objects, data_prefix)
    width, height = slide_size(show)
    return Presentation(slides=reader.slides(show), width=width, height=height)


def slide_size(show: IWAObject) -> tuple[float, float]:
    """Read the size a presentation's slides are laid out at, in points."""
    raw = safe_fields(show.payload).get(SHOW_SLIDE_SIZE_FIELD, [None])[0]
    size = read_point(raw)
    if size is None or size[0] <= 0 or size[1] <= 0:
        return DEFAULT_SLIDE_WIDTH, DEFAULT_SLIDE_HEIGHT
    return size


def titled(block: Block, label: DocItemLabel) -> Block:
    """Relabel a paragraph by where it sits on the slide, not by its style name.

    A Keynote theme names its paragraph styles in the language it was written
    in, so a style called "Title" is not something a reader can rely on seeing.
    The placeholder is, and it is what decides here; a list item is left alone,
    since it is already labelled by the marker it carries.

    Args:
        block: One block read out of a placeholder or a drawable.
        label: The label to give it, if it is a paragraph.

    Returns:
        The block, relabelled where that means anything.
    """
    if not isinstance(block, Paragraph) or block.list_label is not None:
        return block
    return block._replace(label=label, level=None)


class KeynoteReader(IWAReader):
    """Reads the parts of a presentation that ``KN`` adds to the shared ones.

    Everything on a slide is a drawable the shared reader already understands,
    so what is left here is the walk down to them: the show's slide tree, each
    slide's placeholders, its presenter notes, and the sticky notes its comments
    are drawn as. Charts are the one drawable read here rather than there, so
    that only a presentation's are, until another app's are verified too.
    """

    def slides(self, show: IWAObject) -> list[Slide]:
        """Read every slide of the show, in presentation order.

        Args:
            show: The ``KN.ShowArchive`` of the presentation.

        Returns:
            One slide per node of the show's slide tree.
        """
        tree = safe_fields(show.payload).get(SHOW_SLIDE_TREE_FIELD, [None])[0]
        if not isinstance(tree, bytes):
            return []

        slides = []
        for node in iwa_reference_list(tree, SLIDE_TREE_NODES_FIELD):
            slide = self._slide(node)
            if slide is not None:
                slides.append(slide)
        return slides

    def _slide(self, node_id: int) -> Slide | None:
        """Read the slide one node of the slide tree stands for."""
        node = self._objects.get(node_id)
        if node is None or node.message_type != KN_SLIDE_NODE_ARCHIVE:
            return None

        target = iwa_reference_field(node.payload, SLIDE_NODE_SLIDE_FIELD)
        slide = self._objects.get(target) if target is not None else None
        if slide is None or slide.message_type != KN_SLIDE_ARCHIVE:
            return None

        title = iwa_reference_field(slide.payload, SLIDE_TITLE_FIELD)
        number = iwa_reference_field(slide.payload, SLIDE_NUMBER_FIELD)

        placed = self._placed(slide, number)
        blocks: list[Placed] = []
        comments: list[Comment] = []
        for position in reading_order([geometry for _, geometry in placed]):
            identifier, geometry = placed[position]
            found, said = self._blocks(identifier, title=identifier == title)
            blocks.extend(Placed(block, geometry) for block in found)
            comments.extend(said)

        return Slide(blocks=blocks, notes=self._notes(slide), comments=comments)

    def _placed(
        self, slide: IWAObject, number: int | None
    ) -> list[tuple[int, Geometry | None]]:
        """Collect what is on a slide, with where each of it sits.

        Args:
            slide: The ``KN.SlideArchive`` to read.
            number: Its slide-number placeholder, which is left out.

        Returns:
            Each drawable's identifier and geometry, in stored order, with the
            repeats between the slide's two accounts of them dropped.
        """
        identifiers: list[int] = []
        for field in (SLIDE_TITLE_FIELD, SLIDE_BODY_FIELD):
            target = iwa_reference_field(slide.payload, field)
            if target is not None:
                identifiers.append(target)
        identifiers.extend(iwa_reference_list(slide.payload, SLIDE_DRAWABLES_FIELD))

        placed: list[tuple[int, Geometry | None]] = []
        seen: set[int] = set()
        for identifier in identifiers:
            if identifier == number or identifier in seen:
                continue
            seen.add(identifier)
            drawable = self._objects.get(identifier)
            geometry = (
                drawable_geometry(drawable.payload) if drawable is not None else None
            )
            placed.append((identifier, geometry))
        return placed

    def _blocks(
        self, identifier: int, *, title: bool
    ) -> tuple[list[Block], list[Comment]]:
        """Read whichever kind of drawable ``identifier`` names on a slide.

        Args:
            identifier: The drawable to read.
            title: Whether it is the slide's title placeholder.

        Returns:
            The blocks it contributes to the slide, and the comments it holds.
        """
        drawable = self._objects.get(identifier)
        if drawable is None:
            return [], []

        if drawable.message_type == KN_PLACEHOLDER_ARCHIVE:
            if identifier in self._emitted:
                return [], []
            self._emitted.add(identifier)
            label = DocItemLabel.TITLE if title else DocItemLabel.TEXT
            return [
                titled(block, label) for block in self._placeholder_blocks(drawable)
            ], []

        if drawable.message_type == KN_COMMENT_ARCHIVE:
            if identifier in self._emitted:
                return [], []
            self._emitted.add(identifier)
            return [], self._comments(drawable)

        return [
            titled(block, DocItemLabel.TEXT)
            for block in self._drawable_blocks(identifier)
        ], []

    def _drawable_blocks(self, identifier: int) -> list[Block]:
        """Read a drawable, reading a chart here and deferring anything else.

        The shared reader descends into groups through this method too, so a
        chart grouped with other shapes on a slide is found as well.
        """
        drawable = self._objects.get(identifier)
        if drawable is None or drawable.message_type != TSCH_CHART_DRAWABLE:
            return super()._drawable_blocks(identifier)

        if identifier in self._emitted:
            return []
        self._emitted.add(identifier)
        chart = iwa_chart(drawable, self._objects)
        return [chart] if chart is not None else []

    def _placeholder_blocks(self, placeholder: IWAObject) -> list[Block]:
        """Read the text of one placeholder, through the shape info it wraps."""
        shape = safe_fields(placeholder.payload).get(PLACEHOLDER_SHAPE_FIELD, [None])[0]
        if not isinstance(shape, bytes):
            return []

        target = iwa_reference_field(shape, SHAPE_STORAGE_FIELD)
        storage = self._objects.get(target) if target is not None else None
        if storage is None or storage.message_type != TSWP_STORAGE_ARCHIVE:
            return []
        return self.storage_blocks(storage)

    def _comments(self, shape: IWAObject) -> list[Comment]:
        """Read the comment a sticky note on the slide stands for, and its replies."""
        head = iwa_reference_field(shape.payload, COMMENT_STORAGE_FIELD)
        return [Comment(text, "") for text in self._thread(head)]

    def _notes(self, slide: IWAObject) -> list[Paragraph]:
        """Read the presenter notes of one slide."""
        target = iwa_reference_field(slide.payload, SLIDE_NOTE_FIELD)
        note = self._objects.get(target) if target is not None else None
        if note is None or note.message_type != KN_NOTE_ARCHIVE:
            return []
        return self._storage_paragraphs(
            iwa_reference_field(note.payload, NOTE_TEXT_FIELD)
        )
