# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
from io import BytesIO

from docling_core.types.doc import (
    ContentLayer,
    DocItem,
    DoclingDocument,
    GroupLabel,
    PictureItem,
    TextItem,
)

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
    DeclarativeDocumentBackend,
)
from docling.backend.noop_backend import NoOpBackend
from docling.backend.utils.media import MediaKind, get_media_meta, media_file_name
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

_log = logging.getLogger(__name__)


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

        if self.pipeline_options.do_media_conversion:
            with TimeRecorder(conv_res, "media_convert", scope=ProfilingScope.DOCUMENT):
                self._convert_media(conv_res, backend)
        return conv_res

    def _convert_media(
        self, conv_res: ConversionResult, backend: DeclarativeDocumentBackend
    ) -> None:
        """Convert the video and audio files that the document plays.

        The backend records each media file on the picture that shows it, and
        loads the file again on request. The audio or video pipeline converts
        it, and its document goes in a group right after that picture.
        """
        doc = conv_res.document
        for picture in list(doc.pictures):
            media = get_media_meta(picture)
            if media is None:
                continue
            kind, location = media
            try:
                data = backend.load_media(location)
            except OperationNotAllowed as exc:
                # The fetch rules of the backend options decide this, as for images.
                _log.warning("Media file %s is not loaded: %s", location, exc)
                continue
            except Exception as exc:
                self._add_media_error(conv_res, location, f"cannot be loaded: {exc}")
                continue
            if not data:
                continue

            try:
                pipeline = self._get_media_pipeline(kind)
            except ImportError as exc:
                # Like a missing ffmpeg, a missing ASR extra fails the media only.
                self._add_media_error(conv_res, location, f"cannot be converted: {exc}")
                continue
            media_input = InputDocument(
                path_or_stream=BytesIO(data),
                format=InputFormat.VIDEO if kind == "video" else InputFormat.AUDIO,
                backend=NoOpBackend,
                filename=media_file_name(location),
                limits=conv_res.input.limits,
            )
            result = pipeline.execute(media_input, raises_on_error=False)
            if result.status not in {
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            }:
                reasons = "; ".join(error.error_message for error in result.errors)
                self._add_media_error(
                    conv_res, location, f"cannot be converted: {reasons}"
                )
                continue
            self._add_media_document(doc, picture, location, result.document)

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
        picture: PictureItem,
        location: str,
        media_doc: DoclingDocument,
    ) -> None:
        """Add the converted media after its picture.

        The items take the content layer of the picture, so the media of a
        hidden shape stays hidden, and its provenance: the media plays in the
        area of the picture on the slide.
        """
        group = doc.insert_group(
            sibling=picture,
            label=GroupLabel.SECTION,
            name=f"media: {location}",
            content_layer=picture.content_layer,
        )
        doc.add_document(media_doc, parent=group)
        for item, _ in doc.iterate_items(
            root=group,
            with_groups=True,
            traverse_pictures=True,
            included_content_layers=set(ContentLayer),
        ):
            item.content_layer = picture.content_layer
            if isinstance(item, DocItem):
                length = len(item.text) if isinstance(item, TextItem) else 0
                item.prov = [
                    prov.model_copy(update={"charspan": (0, length)})
                    for prov in picture.prov
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
