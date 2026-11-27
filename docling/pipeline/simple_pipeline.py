# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
from io import BytesIO
from pathlib import Path

from docling_core.types.doc import (
    ContentLayer,
    DocItem,
    DoclingDocument,
    MediaRef,
    TextItem,
    TrackItem,
)

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
    DeclarativeDocumentBackend,
)
from docling.backend.noop_backend import NoOpBackend
from docling.backend.utils.media import (
    MediaKind,
    media_file_name,
    media_location,
    track_media,
)
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    InputFormat,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    AsrPipelineOptions,
    ConvertPipelineOptions,
    VideoPipelineOptions,
)
from docling.exceptions import OperationNotAllowed
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.pipeline.base_pipeline import BasePipeline, ConvertPipeline
from docling.pipeline.video_pipeline import VideoPipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder
from docling.utils.video_frame_sampling import (
    ffmpeg_demuxer_for,
    unsupported_container_message,
)

_log = logging.getLogger(__name__)


def _unwrap_track(media_doc: DoclingDocument) -> None:
    """Replace the one track of a converted media file by its content.

    The audio and video pipelines put the transcript and frames in a track; they
    go in the track of the slide instead, as a track has no tracks.
    """
    children = media_doc.body.children
    if len(children) != 1 or not isinstance(
        track := children[0].resolve(media_doc), TrackItem
    ):
        return
    body_ref = media_doc.body.get_ref()
    for ref in track.children:
        ref.resolve(media_doc).parent = body_ref
    media_doc.body.children = [track.get_ref(), *track.children]
    track.children = []
    media_doc.delete_items(node_items=[track])


class SimplePipeline(ConvertPipeline):
    """SimpleModelPipeline.

    This class is used at the moment for formats / backends
    which produce straight DoclingDocument output.
    """

    def __init__(self, pipeline_options: ConvertPipelineOptions):
        super().__init__(pipeline_options)
        # Built on first use, so a document without media loads no ASR model.
        self._media_pipelines: dict[MediaKind, BasePipeline] = {}

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        backend = conv_res.input._backend
        if not isinstance(backend, DeclarativeDocumentBackend):
            raise RuntimeError(
                f"The selected backend {type(backend).__name__} for {conv_res.input.file} is not a declarative backend. "
                f"Can not convert this with simple pipeline. "
                f"Please check your format configuration on DocumentConverter."
            )
            # conv_res.status = ConversionStatus.FAILURE
            # return conv_res

        # Instead of running a page-level pipeline to build up the document structure,
        # the backend is expected to be of type DeclarativeDocumentBackend, which can output
        # a DoclingDocument straight.
        with TimeRecorder(conv_res, "doc_build", scope=ProfilingScope.DOCUMENT):
            conv_res.document = backend.convert()

        options = self.pipeline_options
        if options.keep_media or options.do_media_conversion:
            with TimeRecorder(conv_res, "media_convert", scope=ProfilingScope.DOCUMENT):
                self._process_media(conv_res, backend)
        return conv_res

    def _process_media(
        self, conv_res: ConversionResult, backend: DeclarativeDocumentBackend
    ) -> None:
        """Keep or convert the video and audio files that the document plays.

        The backend turns each media object into a track that references its
        file, and loads the file again on request. With ``keep_media`` the track
        holds the file. With ``do_media_conversion`` the audio or video pipeline
        converts it, and its transcript and frames go in the track.
        """
        doc = conv_res.document
        options = self.pipeline_options
        # Several slides can show the same media part: load and convert each file once.
        loaded: dict[str, bytes | None] = {}
        converted: dict[str, DoclingDocument | None] = {}
        for track in list(doc.tracks):
            if (media := track_media(track)) is None:
                continue
            kind, ref = media
            location = media_location(ref)
            convertible = options.do_media_conversion and self._is_convertible(
                kind, location
            )
            if not (options.keep_media or convertible):
                continue
            if location not in loaded:
                loaded[location] = self._load_media_file(conv_res, backend, location)
            if (data := loaded[location]) is None:
                continue
            if options.keep_media:
                held = MediaRef.from_bytes(data, mimetype=ref.mimetype, uri=ref.uri)
                setattr(track, kind, held)
            if convertible:
                if location not in converted:
                    converted[location] = self._convert_media_file(
                        conv_res, kind, location, data
                    )
                if (media_doc := converted[location]) is not None:
                    self._add_media_document(doc, track, media_doc)

    @staticmethod
    def _is_convertible(kind: MediaKind, location: str) -> bool:
        name = media_file_name(location)
        if kind == "video" and ffmpeg_demuxer_for(Path(name)) is None:
            _log.warning(
                "Media file %s is not converted: %s",
                location,
                unsupported_container_message(Path(name)),
            )
            return False
        return True

    def _load_media_file(
        self,
        conv_res: ConversionResult,
        backend: DeclarativeDocumentBackend,
        location: str,
    ) -> bytes | None:
        """Load one media file; return None when it is not loaded."""
        try:
            data = backend.load_media(location)
        except OperationNotAllowed as exc:
            # The fetch rules of the backend options decide this, as for images.
            _log.warning("Media file %s is not loaded: %s", location, exc)
            return None
        except Exception as exc:
            self._add_media_error(conv_res, location, f"cannot be loaded: {exc}")
            return None
        return data or None

    def _convert_media_file(
        self,
        conv_res: ConversionResult,
        kind: MediaKind,
        location: str,
        data: bytes,
    ) -> DoclingDocument | None:
        """Convert one media file; return None when it is not converted."""
        name = media_file_name(location)
        try:
            pipeline = self._get_media_pipeline(kind)
        except ImportError as exc:
            # Like a missing ffmpeg, a missing ASR extra fails the media only.
            self._add_media_error(conv_res, location, f"cannot be converted: {exc}")
            return None
        media_input = InputDocument(
            path_or_stream=BytesIO(data),
            format=InputFormat.VIDEO if kind == "video" else InputFormat.AUDIO,
            backend=NoOpBackend,
            filename=name,
            limits=conv_res.input.limits,
        )
        result = pipeline.execute(media_input, raises_on_error=False)
        if result.status not in {
            ConversionStatus.SUCCESS,
            ConversionStatus.PARTIAL_SUCCESS,
        }:
            reasons = "; ".join(error.error_message for error in result.errors)
            self._add_media_error(conv_res, location, f"cannot be converted: {reasons}")
            return None
        if result.status == ConversionStatus.PARTIAL_SUCCESS:
            if result.errors:
                conv_res.errors.extend(
                    error.model_copy(
                        update={
                            "error_message": (
                                f"Media file {location}: {error.error_message}"
                            ),
                        }
                    )
                    for error in result.errors
                )
            else:
                self._add_media_error(
                    conv_res, location, "was only partially converted"
                )
        media_doc = result.document
        _unwrap_track(media_doc)
        return media_doc

    def _get_media_pipeline(self, kind: MediaKind) -> BasePipeline:
        if kind not in self._media_pipelines:
            options = self.pipeline_options
            pipeline: BasePipeline
            if kind == "video":
                pipeline = VideoPipeline(
                    VideoPipelineOptions(
                        asr_options=options.media_asr_options,
                        accelerator_options=options.accelerator_options,
                        artifacts_path=options.artifacts_path,
                        document_timeout=options.document_timeout,
                    )
                )
            else:
                pipeline = AsrPipeline(
                    AsrPipelineOptions(
                        asr_options=options.media_asr_options,
                        accelerator_options=options.accelerator_options,
                        artifacts_path=options.artifacts_path,
                        document_timeout=options.document_timeout,
                    )
                )
            self._media_pipelines[kind] = pipeline
        return self._media_pipelines[kind]

    @staticmethod
    def _add_media_document(
        doc: DoclingDocument,
        track: TrackItem,
        media_doc: DoclingDocument,
    ) -> None:
        """Add the transcript and frames of the converted media to its track.

        The items take the content layer of the track, so the media of a hidden
        shape stays hidden, and its provenance: the media plays in the area of
        the track on the slide.
        """
        first = len(track.children)
        doc.add_document(media_doc, parent=track)
        for child in track.children[first:]:
            for item, _ in doc.iterate_items(
                root=child.resolve(doc),
                with_groups=True,
                traverse_pictures=True,
                included_content_layers=set(ContentLayer),
            ):
                item.content_layer = track.content_layer
                if isinstance(item, DocItem):
                    length = len(item.text) if isinstance(item, TextItem) else 0
                    item.prov = [
                        prov.model_copy(update={"charspan": (0, length)})
                        for prov in track.prov
                    ]

    @staticmethod
    def _add_media_error(
        conv_res: ConversionResult, location: str, message: str
    ) -> None:
        _log.warning("Media file %s %s", location, message)
        conv_res.errors.append(
            ErrorItem(
                component_type=DoclingComponentType.PIPELINE,
                module_name=SimplePipeline.__name__,
                error_message=f"Media file {location} {message}",
            )
        )

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        # This is called only if the previous steps didn't raise.
        # Since we don't have anything else to evaluate, we can
        # safely return SUCCESS.
        return ConversionStatus.SUCCESS

    @classmethod
    def get_default_options(cls) -> ConvertPipelineOptions:
        return ConvertPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, DeclarativeDocumentBackend)
