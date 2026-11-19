# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Map native GitHub conversations to structured Docling documents."""

import codecs
import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import cast
from urllib.parse import quote, urljoin, urlsplit

from docling_core.types.doc import (
    BaseMeta,
    CodeItem,
    CodeLanguageLabel,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    FloatingItem,
    FloatingMeta,
    GroupItem,
    ImageRef,
    InlineGroup,
    NodeItem,
    RefItem,
    RichTableCell,
    TableCell,
    TableData,
    TableItem,
    TextItem,
)
from pydantic import AnyUrl
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.utils.image_resource_loader import ImageResourceLoader
from docling.datamodel.backend_options import (
    BackendOptions,
    GitHubJSONBackendOptions,
    MarkdownBackendOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.github import (
    MAX_GITHUB_JSON_BYTES,
    GitHubConversation,
    GitHubEntry,
    parse_github_json,
)

_log = logging.getLogger(__name__)


class _CachedImageLoader(ImageResourceLoader):
    def __init__(self, options: GitHubJSONBackendOptions) -> None:
        super().__init__(
            enable_local_fetch=options.enable_local_fetch,
            enable_remote_fetch=options.enable_remote_fetch,
            max_image_data_base64_bytes=options.max_image_data_base64_bytes,
            max_image_pixels=options.max_image_pixels,
        )
        self.images: dict[tuple[str, str | None], ImageRef | None] = {}

    @override
    def load_image_ref(self, src: str, base_path: str | None) -> ImageRef | None:
        key = (src, base_path)
        if key not in self.images:
            self.images[key] = super().load_image_ref(src, base_path)
        return self.images[key]


class GitHubJSONBackend(DeclarativeDocumentBackend):
    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: BackendOptions | None = None,
    ) -> None:
        selected = (
            options
            if isinstance(options, GitHubJSONBackendOptions)
            else GitHubJSONBackendOptions()
        )
        super().__init__(in_doc, path_or_stream, selected)
        self._options = selected
        self._images = _CachedImageLoader(selected)
        self._conversation: GitHubConversation | None = None
        self._error: Exception | None = None
        try:
            if isinstance(path_or_stream, Path):
                with path_or_stream.open("rb") as handle:
                    content = handle.read(MAX_GITHUB_JSON_BYTES + 1)
            else:
                content = path_or_stream.getvalue()
            if len(content) > MAX_GITHUB_JSON_BYTES:
                raise ValueError("GitHub JSON exceeds the 32 MiB input limit")
            self._conversation = parse_github_json(
                json.loads(content.removeprefix(codecs.BOM_UTF8))
            )
        except (ValueError, OSError) as exc:
            self._error = exc

    @override
    def is_valid(self) -> bool:
        return self._conversation is not None

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.JSON_GITHUB}

    def _resolve_link(
        self, target: str, conversation: GitHubConversation
    ) -> AnyUrl | Path:
        if urlsplit(target).scheme:
            return AnyUrl(target)
        if target.startswith(("#", "/", "?")):
            return AnyUrl(urljoin(conversation.source.url, target))
        if conversation.repository_ref is not None:
            base = (
                f"https://github.com/{conversation.source.owner}/{conversation.source.repository}"
                f"/blob/{quote(conversation.repository_ref, safe='')}/"
            )
            return AnyUrl(urljoin(base, target))
        _log.warning(
            "Kept unresolved repository-relative link %r: no repository ref was supplied",
            target,
        )
        return Path(target)

    def _append_markdown(
        self,
        doc: DoclingDocument,
        parent: GroupItem,
        body: str,
        conversation: GitHubConversation,
    ) -> None:
        if not body:
            return
        # Keep the Markdown/HTML dependencies optional until conversion is requested.
        try:
            from docling.backend.json.github_markdown import GitHubMarkdownBackend
        except ModuleNotFoundError as exc:
            if exc.name in {"marko", "bs4"}:
                raise ImportError(
                    "GitHub JSON conversion requires `pip install 'docling-slim[format-web]'`."
                ) from exc
            raise

        options = MarkdownBackendOptions.model_validate(
            self._options.model_dump(exclude={"strict_assets"})
        )
        options.source_uri = self._options.source_uri or (
            self.path_or_stream
            if isinstance(self.path_or_stream, Path)
            and self._options.enable_local_fetch
            else AnyUrl(conversation.source.url)
            if conversation.repository_ref is not None
            else None
        )
        stream = BytesIO(body.encode("utf-8"))
        in_doc = InputDocument(
            path_or_stream=stream,
            filename=f"{self.file.stem}.md",
            format=InputFormat.MD,
            backend=GitHubMarkdownBackend,
            backend_options=options,
        )
        backend = cast(GitHubMarkdownBackend, in_doc._backend)
        backend.link_resolver = lambda target: (
            str(self._resolve_link(target, conversation)) if target else target
        )
        # Share the loader across entries so repeated Markdown images are fetched once.
        backend._image_loader = self._images
        fragment = backend.convert()
        for item in list(fragment.texts):
            if isinstance(item.hyperlink, Path):
                original = str(item.hyperlink)
                item.hyperlink = self._resolve_link(original, conversation)
                if item.meta is None:
                    item.meta = (
                        FloatingMeta() if isinstance(item, FloatingItem) else BaseMeta()
                    )
                item.meta.set_custom_field("github", "original_link", original)
            if (
                item.hyperlink is not None
                and not isinstance(item, CodeItem)
                and item.parent is not None
                and isinstance(item.parent.resolve(fragment), InlineGroup)
                and item.text.strip()
            ):
                # The HTML path can move surrounding whitespace into an anchor.
                # Keep the rendered text, but put that whitespace outside its link.
                text = item.text
                leading = text[: len(text) - len(text.lstrip())]
                trailing = text[len(text.rstrip()) :]
                item.text = text.strip()
                item.orig = item.text
                if leading:
                    fragment.insert_text(
                        sibling=item, label=DocItemLabel.TEXT, text=leading, after=False
                    )
                if trailing:
                    fragment.insert_text(
                        sibling=item, label=DocItemLabel.TEXT, text=trailing
                    )
        missing = sum(picture.image is None for picture in fragment.pictures)
        if missing:
            if self._options.strict_assets:
                raise ValueError(f"Could not load {missing} GitHub picture asset(s)")
            _log.warning(
                "GitHub entry contains %d unavailable picture asset(s)", missing
            )
        self._append_fragment(doc, parent, fragment)

    @staticmethod
    def _append_fragment(
        doc: DoclingDocument, parent: GroupItem, fragment: DoclingDocument
    ) -> None:
        originals = [ref.resolve(fragment) for ref in fragment.body.children]
        start = len(parent.children)
        doc.add_node_items(node_items=originals, doc=fragment, parent=parent)
        copies = [ref.resolve(doc) for ref in parent.children[start:]]
        pending: list[tuple[NodeItem, NodeItem]] = list(
            zip(originals, copies, strict=True)
        )
        pairs: list[tuple[NodeItem, NodeItem]] = []
        lookup: dict[str, RefItem] = {}
        while pending:
            original, copied = pending.pop()
            lookup[original.self_ref] = copied.get_ref()
            pairs.append((original, copied))
            pending.extend(
                (old.resolve(fragment), new.resolve(doc))
                for old, new in zip(original.children, copied.children, strict=True)
            )
        # add_node_items copies the tree but does not rebase explicit rich-cell
        # and floating-item references. Map those using the actual copied nodes.
        for original, copied in pairs:
            if isinstance(original, TableItem) and isinstance(copied, TableItem):
                for old, new in zip(
                    original.data.table_cells, copied.data.table_cells, strict=True
                ):
                    if isinstance(old, RichTableCell) and isinstance(
                        new, RichTableCell
                    ):
                        new.ref = lookup[old.ref.cref]
            if isinstance(original, FloatingItem) and isinstance(copied, FloatingItem):
                copied.captions = [lookup[ref.cref] for ref in original.captions]
                copied.footnotes = [lookup[ref.cref] for ref in original.footnotes]
                copied.references = [lookup[ref.cref] for ref in original.references]

    @staticmethod
    def _metadata(item: TextItem, name: str, value: object) -> None:
        if item.meta is None:
            item.meta = FloatingMeta() if isinstance(item, FloatingItem) else BaseMeta()
        item.meta.set_custom_field("github", name, value)

    @staticmethod
    def _ordered(entries: list[GitHubEntry]) -> list[GitHubEntry]:
        def timestamp(entry: GitHubEntry) -> datetime:
            value = entry.created_at or datetime.min.replace(tzinfo=timezone.utc)
            return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value

        return sorted(entries, key=timestamp)

    def _append_entry(
        self,
        doc: DoclingDocument,
        parent: GroupItem,
        entry: GitHubEntry,
        conversation: GitHubConversation,
    ) -> None:
        group = doc.add_group(
            name=f"github:{entry.kind}:{entry.id or entry.url or len(doc.groups)}",
            parent=parent,
        )
        title = entry.kind.replace("_", " ").title()
        title += f" by {entry.author or '[deleted]'}"
        if entry.is_answer:
            title += " (Accepted answer)"
        header = doc.add_heading(
            text=title,
            level=2,
            parent=group,
            hyperlink=AnyUrl(entry.url) if entry.url else None,
        )
        self._metadata(
            header, "entry", entry.model_dump(mode="json", exclude={"replies"})
        )
        details = [
            ("Created", entry.created_at),
            ("Updated", entry.updated_at),
            ("State", entry.state),
            ("File", entry.path),
            ("Line", entry.line),
            ("Start line", entry.start_line),
            ("Original line", entry.original_line),
            ("Resolved", entry.is_resolved),
            ("Outdated", entry.is_outdated),
        ]
        for key, value in details:
            if value is not None:
                doc.add_text(
                    label=DocItemLabel.TEXT, text=f"{key}: {value}", parent=group
                )
        if entry.diff_hunk is not None:
            snippet = doc.add_code(
                text=entry.diff_hunk,
                code_language=CodeLanguageLabel.UNKNOWN,
                parent=group,
            )
            self._metadata(snippet, "language", "diff")
        self._append_markdown(doc, group, entry.body, conversation)
        if not entry.body and not entry.replies:
            doc.add_text(
                label=DocItemLabel.TEXT, text="No body supplied.", parent=group
            )
        for reply in self._ordered(entry.replies):
            self._append_entry(doc, group, reply, conversation)

    def _append_labels(
        self, doc: DoclingDocument, conversation: GitHubConversation
    ) -> None:
        group = doc.add_group(name="github:labels")
        doc.add_heading(text="Labels", level=1, parent=group)
        if not conversation.labels:
            doc.add_text(
                label=DocItemLabel.TEXT,
                text="Not supplied" if conversation.labels is None else "None",
                parent=group,
            )
            return
        table = doc.add_table(
            data=TableData(num_rows=len(conversation.labels) + 1, num_cols=2),
            parent=group,
        )
        for col, text in enumerate(("Name", "Description")):
            doc.add_table_cell(
                table,
                TableCell(
                    text=text,
                    column_header=True,
                    start_row_offset_idx=0,
                    end_row_offset_idx=1,
                    start_col_offset_idx=col,
                    end_col_offset_idx=col + 1,
                ),
            )
        for row, label in enumerate(conversation.labels, start=1):
            inline = doc.add_inline_group(parent=table)
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=label.name,
                parent=inline,
                hyperlink=AnyUrl(label.url) if label.url else None,
            )
            doc.add_table_cell(
                table,
                RichTableCell(
                    text=label.name,
                    ref=inline.get_ref(),
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + 1,
                    start_col_offset_idx=0,
                    end_col_offset_idx=1,
                ),
            )
            doc.add_table_cell(
                table,
                TableCell(
                    text=label.description or "",
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + 1,
                    start_col_offset_idx=1,
                    end_col_offset_idx=2,
                ),
            )

    @override
    def convert(self) -> DoclingDocument:
        if self._error is not None:
            raise self._error
        if self._conversation is None:
            raise ValueError("No valid GitHub conversation was loaded")
        conversation = self._conversation
        doc = DoclingDocument(
            name=f"{conversation.source.repository}-{conversation.source.kind}-{conversation.source.number}",
            origin=DocumentOrigin(
                filename=self.file.name,
                mimetype="application/json",
                binary_hash=int(self.document_hash, 16) & 0xFFFFFFFFFFFFFFFF,
            ),
        )
        title = doc.add_title(
            text=conversation.title, hyperlink=AnyUrl(conversation.source.url)
        )
        self._metadata(
            title, "source", conversation.model_dump(mode="json", exclude={"entries"})
        )
        source = doc.add_group(name="github:source")
        doc.add_heading(text="Source information", level=1, parent=source)
        details = [
            (
                "Repository",
                f"{conversation.source.owner}/{conversation.source.repository}",
            ),
            ("Type", conversation.source.kind),
            ("Number", conversation.source.number),
            ("Author", conversation.author or "[deleted]"),
            ("State", conversation.state),
            ("Created", conversation.created_at),
            ("Updated", conversation.updated_at),
            ("Category", conversation.category),
            ("Accepted answer ID", conversation.accepted_answer_id),
            (
                "Assignees",
                ", ".join(conversation.assignees)
                if conversation.assignees is not None
                else None,
            ),
            (
                "Retrieval complete",
                conversation.retrieval_complete
                if conversation.retrieval_complete is not None
                else "Unknown (supplied JSON)",
            ),
        ]
        for key, value in details:
            if value is not None:
                doc.add_text(
                    label=DocItemLabel.TEXT, text=f"{key}: {value}", parent=source
                )
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=conversation.source.url,
            hyperlink=AnyUrl(conversation.source.url),
            parent=source,
        )
        if conversation.missing_collections:
            doc.add_text(
                label=DocItemLabel.TEXT,
                text="Not supplied: " + ", ".join(conversation.missing_collections),
                parent=source,
            )
        if conversation.source.kind != "discussion":
            self._append_labels(doc, conversation)
        body = doc.add_group(name="github:body")
        doc.add_heading(text="Body", level=1, parent=body)
        self._append_markdown(doc, body, conversation.body, conversation)
        if not conversation.body:
            doc.add_text(label=DocItemLabel.TEXT, text="No body supplied.", parent=body)
        if conversation.entries:
            entries = doc.add_group(name="github:conversation")
            doc.add_heading(text="Conversation", level=1, parent=entries)
            for entry in self._ordered(conversation.entries):
                self._append_entry(doc, entries, entry, conversation)
        return doc
