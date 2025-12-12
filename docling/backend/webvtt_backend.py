import copy
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Union

from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    Formatting,
    ProvenanceTrack,
)
from docling_core.types.doc.webvtt import (
    WebVTTCueBoldSpan,
    WebVTTCueComponent,
    WebVTTCueComponentWithTerminator,
    WebVTTCueItalicSpan,
    WebVTTCueLanguageSpan,
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
    voice: Optional[str] = None
    formatting: Optional[Formatting] = None
    classes: dict[Literal["b", "u", "i", "lang", "v"], set[str]] = field(
        default_factory=dict
    )
    lang: set[str] = field(default_factory=set)

    def copy_meta(self, text):
        return AnnotatedText(
            text=text,
            voice=self.voice,
            formatting=self.formatting.model_copy() if self.formatting else None,
            classes=copy.deepcopy(self.classes),
            lang=self.lang.copy(),
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
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]):
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
        parent_idx: int = -1

        def _extract_component_text(
            payload: list[WebVTTCueComponentWithTerminator],
        ) -> None:
            nonlocal cue_text, parent_idx
            if not cue_text:
                cue_text.append(AnnotatedPar(items=[]))
            par = cue_text[-1]
            for comp in payload:
                if not par.items:
                    par.items.append(AnnotatedText(text=""))
                    parent_idx = 0
                parent_item: AnnotatedText = par.items[parent_idx]
                current_item: AnnotatedText
                if parent_item.text:
                    current_item = parent_item.copy_meta("")
                    par.items.append(current_item)
                else:
                    current_item = parent_item
                if isinstance(comp.component, WebVTTCueTextSpan):
                    current_item.text += comp.component.text
                elif isinstance(comp.component, WebVTTCueBoldSpan):
                    if not current_item.formatting:
                        current_item.formatting = Formatting(bold=True)
                    else:
                        current_item.formatting.bold = True
                    current_item.classes.setdefault("b", set()).update(
                        comp.component.start_tag.classes
                    )
                    current_parent = parent_idx
                    parent_idx = len(par.items) - 1
                    _extract_component_text(comp.component.internal_text.components)
                    parent_idx = current_parent
                elif isinstance(comp.component, WebVTTCueItalicSpan):
                    if not current_item.formatting:
                        current_item.formatting = Formatting(italic=True)
                    else:
                        current_item.formatting.italic = True
                    current_item.classes.setdefault("i", set()).update(
                        comp.component.start_tag.classes
                    )
                    current_parent = parent_idx
                    parent_idx = len(par.items) - 1
                    _extract_component_text(comp.component.internal_text.components)
                    parent_idx = current_parent
                elif isinstance(comp.component, WebVTTCueUnderlineSpan):
                    if not current_item.formatting:
                        current_item.formatting = Formatting(underline=True)
                    else:
                        current_item.formatting.underline = True
                    current_item.classes.setdefault("u", set()).update(
                        comp.component.start_tag.classes
                    )
                    current_parent = parent_idx
                    parent_idx = len(par.items) - 1
                    _extract_component_text(comp.component.internal_text.components)
                    parent_idx = current_parent
                elif isinstance(comp.component, WebVTTCueLanguageSpan):
                    current_item.lang.add(comp.component.start_tag.annotation)
                    current_item.classes.setdefault("lang", set()).update(
                        comp.component.start_tag.classes
                    )
                    current_parent = parent_idx
                    parent_idx = len(par.items) - 1
                    _extract_component_text(comp.component.internal_text.components)
                    parent_idx = current_parent
                elif isinstance(comp.component, WebVTTCueVoiceSpan):
                    # voice spans cannot be embedded -> overwrite with the last annotation
                    current_item.voice = comp.component.start_tag.annotation
                    current_parent = parent_idx
                    parent_idx = len(par.items) - 1
                    _extract_component_text(comp.component.internal_text.components)
                    parent_idx = current_parent
                if comp.terminator is not None:
                    cue_text.append(AnnotatedPar(items=[]))
                    par = cue_text[-1]

        for block in vtt.cue_blocks:
            cue_text = []
            parent_idx = -1
            identifier = str(block.identifier) if block.identifier else None
            _extract_component_text(block.payload)
            for par in cue_text:
                if not par.items:
                    continue
                elif len(par.items) == 1:
                    lang: Optional[list[str]] = (
                        list(par.items[0].lang) if par.items[0].lang else None
                    )
                    classes: Optional[list[str]] = None
                    if par.items[0].classes:
                        classes = [
                            ".".join([key, *value])
                            for key, value in par.items[0].classes.items()
                        ]
                    track: ProvenanceTrack = ProvenanceTrack(
                        start_time=block.timings.start,
                        end_time=block.timings.end,
                        identifier=identifier,
                        languages=lang,
                        classes=classes,
                        voice=par.items[0].voice if par.items[0].voice else None,
                    )
                    doc.add_text(
                        label=DocItemLabel.TEXT,
                        text=par.items[0].text,
                        content_layer=ContentLayer.BODY,
                        prov=track,
                        formatting=par.items[0].formatting,
                    )
                else:
                    group = doc.add_inline_group(
                        "WebVTT cue span", content_layer=ContentLayer.BODY
                    )
                    for item in par.items:
                        lang = list(item.lang) if item.lang else None
                        classes = None
                        if item.classes:
                            classes = [
                                ".".join([key, *value])
                                for key, value in item.classes.items()
                            ]
                        track = ProvenanceTrack(
                            start_time=block.timings.start,
                            end_time=block.timings.end,
                            identifier=identifier,
                            languages=lang,
                            classes=classes,
                            voice=item.voice if item.voice else None,
                        )
                        doc.add_text(
                            label=DocItemLabel.TEXT,
                            text=item.text,
                            content_layer=ContentLayer.BODY,
                            prov=track,
                            parent=group,
                            formatting=item.formatting,
                        )

        return doc
