import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    Formatting,
    TrackSource,
)
from docling_core.types.doc.webvtt import (
    WebVTTCueBoldSpan,
    WebVTTCueComponent,
    WebVTTCueComponentWithTerminator,
    WebVTTCueItalicSpan,
    WebVTTCueTextSpan,
    WebVTTCueUnderlineSpan,
    WebVTTCueVoiceSpan,
    WebVTTFile,
)
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


@dataclass
class AnnotatedText:
    text: str
    voice: str | None = None
    formatting: Formatting | None = None

    def copy_meta(self, text):
        return AnnotatedText(
            text=text,
            voice=self.voice,
            formatting=self.formatting.model_copy() if self.formatting else None,
        )


@dataclass
class AnnotatedPar:
    items: list[AnnotatedText]


class WebVTTDocumentBackend(DeclarativeDocumentBackend):
    """Declarative backend for WebVTT (.vtt) files.

    This parser reads the content of a WebVTT file and converts
    it to a DoclingDocument, following the W3C specs on https://www.w3.org/TR/webvtt1

    Each cue becomes a TextItem and the items are appended to the
    document body by the cue's start time.
    """

    @override
    def __init__(self, in_doc: InputDocument, path_or_stream: BytesIO | Path):
        super().__init__(in_doc, path_or_stream)

        self.content: str = ""
        try:
            if isinstance(self.path_or_stream, BytesIO):
                self.content = self.path_or_stream.getvalue().decode("utf-8")
            if isinstance(self.path_or_stream, Path):
                with open(self.path_or_stream, encoding="utf-8") as f:
                    self.content = f.read()
        except Exception as e:
            raise RuntimeError(
                "Could not initialize the WebVTT backend for file with hash "
                f"{self.document_hash}."
            ) from e

    @override
    def is_valid(self) -> bool:
        return WebVTTFile.verify_signature(self.content)

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @override
    def unload(self):
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()
        self.path_or_stream = None

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.VTT}

    @override
    def convert(self) -> DoclingDocument:
        _log.debug("Starting WebVTT conversion...")
        if not self.is_valid():
            raise RuntimeError("Invalid WebVTT document.")

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="text/vtt",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        vtt: WebVTTFile = WebVTTFile.parse(self.content)
        cue_text: list[AnnotatedPar] = []
        parents: list[AnnotatedText] = []

        def _extract_components(
            payload: list[WebVTTCueComponentWithTerminator],
        ) -> None:
            nonlocal cue_text, parents
            if not cue_text:
                cue_text.append(AnnotatedPar(items=[]))
            par = cue_text[-1]
            for comp in payload:
                item: AnnotatedText = (
                    parents[-1].copy_meta("") if parents else AnnotatedText(text="")
                )
                component: WebVTTCueComponent = comp.component
                if isinstance(component, WebVTTCueTextSpan):
                    item.text = component.text
                    par.items.append(item)
                else:
                    # configure metadata based on span type
                    if isinstance(component, WebVTTCueBoldSpan):
                        item.formatting = item.formatting or Formatting()
                        item.formatting.bold = True

                    elif isinstance(component, WebVTTCueItalicSpan):
                        item.formatting = item.formatting or Formatting()
                        item.formatting.italic = True

                    elif isinstance(component, WebVTTCueUnderlineSpan):
                        item.formatting = item.formatting or Formatting()
                        item.formatting.underline = True

                    elif isinstance(component, WebVTTCueVoiceSpan):
                        # voice spans cannot be embedded
                        item.voice = component.start_tag.annotation

                    parents.append(item)
                    _extract_components(component.internal_text.components)
                    parents.pop()

                if comp.terminator is not None:
                    cue_text.append(AnnotatedPar(items=[]))
                    par = cue_text[-1]

        def _add_text_item(
            text: str,
            formatting: Formatting | None,
            item: AnnotatedText,
            parent=None,
        ):
            track = TrackSource(
                start_time=block.timings.start.seconds,
                end_time=block.timings.end.seconds,
                identifier=identifier,
                voice=item.voice or None,
            )

            doc.add_text(
                label=DocItemLabel.TEXT,
                text=text,
                content_layer=ContentLayer.BODY,
                formatting=formatting,
                parent=parent,
                source=track,
            )

        if vtt.title:
            doc.add_title(vtt.title, content_layer=ContentLayer.BODY)
        for block in vtt.cue_blocks:
            cue_text = []
            parents = []
            identifier = str(block.identifier) if block.identifier else None
            _extract_components(block.payload)
            for par in cue_text:
                if not par.items:
                    continue
                if len(par.items) == 1:
                    item = par.items[0]
                    _add_text_item(
                        text=item.text,
                        formatting=item.formatting,
                        item=item,
                    )
                else:
                    group = doc.add_inline_group(
                        "WebVTT cue span", content_layer=ContentLayer.BODY
                    )
                    for item in par.items:
                        _add_text_item(
                            text=item.text,
                            formatting=item.formatting,
                            item=item,
                            parent=group,
                        )

        return doc
