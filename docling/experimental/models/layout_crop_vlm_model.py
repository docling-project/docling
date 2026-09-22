# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Recognize layout regions with a VLM, one crop per region.

The layout model owns the geometry of the page; this stage only fills in the
content. Every region is cropped, sent to the VLM with the task that matches
its layout label, and the DocLang reply is written back into the structures the
page assembler reads: text cells for text-like regions, a table prediction for
tables.
"""

import logging
import re
from collections.abc import Iterable
from pathlib import Path

from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.types.doc import (
    BoundingBox,
    DoclingDocument,
    TableItem,
    TextItem,
)
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image as PILImage

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    Cluster,
    Page,
    Table,
    TableStructurePrediction,
)
from docling.datamodel.document import ConversionResult
from docling.experimental.datamodel.layout_crop_vlm_pipeline_options import (
    CropTask,
    LayoutCropVlmOptions,
)
from docling.models.base_layout_model import TABLE_LABELS, TEXT_ELEM_LABELS
from docling.models.base_model import BasePageModel
from docling.models.inference_engines.vlm import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineType,
    create_vlm_engine,
)
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

_DOCLANG_CLOSE = "</doclang>"
_TAG_RE = re.compile(r"<[^>]+>")
_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)(?:\]\]>|$)", re.DOTALL)


class LayoutCropVlmModel(BasePageModel):
    """Page stage that reads each layout region from its crop with a VLM."""

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Path | str | None,
        options: LayoutCropVlmOptions,
        accelerator_options: AcceleratorOptions,
    ):
        self.enabled = enabled
        self.options = options

        if not self.enabled:
            return

        vlm_options = options.vlm_options
        self.engine: BaseVlmEngine = create_vlm_engine(
            options=vlm_options.engine_options,
            model_spec=vlm_options.model_spec,
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
            enable_remote_services=enable_remote_services,
        )

    def _resolve_runtime_engine_type(self) -> VlmEngineType:
        # The auto-inline engine picks its concrete engine at runtime and is the
        # only one exposing the choice, hence the attribute probe.
        selected_engine_type = getattr(self.engine, "selected_engine_type", None)
        if selected_engine_type is not None:
            return selected_engine_type
        return self.options.vlm_options.engine_options.engine_type

    def _task_for(self, cluster: Cluster) -> CropTask | None:
        task = self.options.tasks.get(cluster.label)
        if task is not None:
            return task
        if cluster.label in TEXT_ELEM_LABELS:
            return self.options.default_task
        return None

    def _crop(self, page: Page, cluster: Cluster) -> PILImage.Image | None:
        assert page.size is not None
        bbox = cluster.bbox
        pad_x = bbox.width * self.options.crop_padding
        pad_y = bbox.height * self.options.crop_padding
        padded = BoundingBox(
            l=max(0.0, bbox.l - pad_x),
            t=max(0.0, bbox.t - pad_y),
            r=min(page.size.width, bbox.r + pad_x),
            b=min(page.size.height, bbox.b + pad_y),
            coord_origin=bbox.coord_origin,
        )
        if padded.width < 1 or padded.height < 1:
            return None
        return page.get_image(scale=self.options.crop_scale, cropbox=padded)

    @staticmethod
    def _parse_reply(text: str) -> DoclingDocument | None:
        start = text.find("<doclang")
        if start < 0:
            return None
        fragment = text[start:]
        # Servers strip the stop string, so the closing tag is usually missing.
        end = fragment.find(_DOCLANG_CLOSE)
        if end < 0:
            fragment += _DOCLANG_CLOSE
        else:
            fragment = fragment[: end + len(_DOCLANG_CLOSE)]
        try:
            return DocLangDocDeserializer().deserialize_str(fragment)
        except Exception as exc:
            _log.debug("DocLang reply did not parse: %s", exc)
            return None

    @staticmethod
    def _salvage_text(text: str) -> str:
        """Best-effort text of a reply that is not well-formed DocLang.

        A reply cut off by the token budget leaves tags open; its text is still
        worth more than an empty region.
        """
        text = _CDATA_RE.sub(lambda m: m.group(1), text)
        return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text)).strip()

    def _apply_text(self, cluster: Cluster, reply: str) -> None:
        doc = self._parse_reply(reply)
        if doc is None:
            text = self._salvage_text(reply)
        else:
            text = "\n".join(
                item.text
                for item, _ in doc.iterate_items()
                if isinstance(item, TextItem) and item.text.strip()
            )
        if not text:
            return
        cluster.cells = [
            TextCell(
                index=0,
                text=text,
                orig=text,
                from_ocr=True,
                rect=BoundingRectangle.from_bounding_box(cluster.bbox),
            )
        ]

    def _apply_table(self, page: Page, cluster: Cluster, reply: str) -> None:
        doc = self._parse_reply(reply)
        table_item = None
        if doc is not None:
            table_item = next(
                (
                    item
                    for item, _ in doc.iterate_items()
                    if isinstance(item, TableItem)
                ),
                None,
            )
        if table_item is None:
            _log.warning(
                "Page %s: no table in the VLM reply for region %s",
                page.page_no,
                cluster.id,
            )
            return
        if page.predictions.tablestructure is None:
            page.predictions.tablestructure = TableStructurePrediction()
        page.predictions.tablestructure.table_map[cluster.id] = Table(
            label=cluster.label,
            id=cluster.id,
            page_no=page.page_no,
            cluster=cluster,
            text="",
            otsl_seq=[],
            num_rows=table_item.data.num_rows,
            num_cols=table_item.data.num_cols,
            table_cells=table_item.data.table_cells,
        )

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        page_list = list(page_batch)

        with TimeRecorder(conv_res, "layout_crop_vlm"):
            model_spec = self.options.vlm_options.model_spec
            stop_strings = list(model_spec.stop_strings)
            extra_generation_config = model_spec.get_runtime_input_extra_config(
                self._resolve_runtime_engine_type()
            )

            targets: list[tuple[Page, Cluster]] = []
            engine_inputs: list[VlmEngineInput] = []
            for page in page_list:
                if page._backend is None or not page._backend.is_valid():
                    continue
                if page.predictions.layout is None:
                    continue
                # Render the page once; the crops are then cut from the cache.
                page.get_image(scale=self.options.crop_scale)
                for cluster in page.predictions.layout.clusters:
                    task = self._task_for(cluster)
                    if task is None:
                        continue
                    image = self._crop(page, cluster)
                    if image is None:
                        continue
                    targets.append((page, cluster))
                    engine_inputs.append(
                        VlmEngineInput(
                            image=image,
                            prompt=task.prompt,
                            response_prefix=(
                                task.response_prefix
                                if self.options.use_response_prefix
                                else None
                            ),
                            temperature=model_spec.temperature,
                            max_new_tokens=self.options.max_new_tokens
                            or model_spec.max_new_tokens,
                            stop_strings=stop_strings,
                            extra_generation_config=extra_generation_config,
                        )
                    )

            batch_size = self.options.engine_batch_size
            for start in range(0, len(engine_inputs), batch_size):
                outputs = self.engine.predict_batch(
                    engine_inputs[start : start + batch_size]
                )
                # Every reply at least echoes its prefix or opens a DocLang element,
                # so an empty one is a failed request. Failing the batch beats
                # assembling pages whose regions are silently blank.
                failed = sum(1 for output in outputs if not output.text)
                if failed:
                    raise RuntimeError(
                        f"The VLM engine returned no reply for {failed} of "
                        f"{len(outputs)} regions."
                    )
                for (page, cluster), output in zip(targets[start:], outputs):
                    _log.debug(
                        "Page %s region %s (%s): %r",
                        page.page_no,
                        cluster.id,
                        cluster.label.value,
                        output.text,
                    )
                    if cluster.label in TABLE_LABELS:
                        self._apply_table(page, cluster, output.text)
                    else:
                        self._apply_text(cluster, output.text)

        yield from page_list

    def __del__(self) -> None:
        engine = self.__dict__.get("engine")
        if engine is not None:
            engine.cleanup()
