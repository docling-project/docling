# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import importlib.metadata
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import torch
from docling_core.types.doc import DocItemLabel
from transformers import AutoModelForImageTextToText, AutoProcessor

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
from docling.models.base_table_model import BaseTableStructureModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device
from docling.utils.granite_vision_utils import granite_vision_4_needs_remote_code
from docling.utils.otsl import parse_otsl_output
from docling.utils.profiling import TimeRecorder
from docling.utils.table_cell_postprocess import (
    consolidate_duplicate_spans,
    promote_wrapped_header_rows,
    split_into_structural_subtables,
)

_log = logging.getLogger(__name__)


class GraniteVisionTableStructureModel(BaseTableStructureModel):
    """Table structure model using ibm-granite/granite-vision-4.1-4b with <tables_otsl>."""

    _model_repo_id: ClassVar[str] = "ibm-granite/granite-vision-4.1-4b"
    _model_repo_folder: ClassVar[str] = "ibm-granite--granite-vision-4.1-4b"
    _model_repo_revision: ClassVar[str] = "dd48e97503de471803850df70843cf9eb5da8712"

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: GraniteVisionTableStructureOptions,
        accelerator_options: AcceleratorOptions,
        enable_remote_services: Literal[False] = False,
    ):
        self.enabled = enabled
        self.options = options
        self.accelerator_options = accelerator_options

        if self.enabled:
            self.device = decide_device(
                accelerator_options.device,
                supported_devices=[AcceleratorDevice.CPU, AcceleratorDevice.CUDA],
            )

            if artifacts_path is None:
                artifacts_path = self.download_models()
            elif (artifacts_path / self._model_repo_folder).exists():
                artifacts_path = artifacts_path / self._model_repo_folder
            else:
                _log.warning(
                    f"Model artifacts not found at {artifacts_path / self._model_repo_folder},"
                    " they will be downloaded."
                )

            self._load_model(artifacts_path)

    @classmethod
    def get_options_type(cls) -> type[GraniteVisionTableStructureOptions]:
        return GraniteVisionTableStructureOptions

    @classmethod
    def download_models(
        cls,
        local_dir: Path | None = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        return download_hf_model(
            repo_id=cls._model_repo_id,
            revision=cls._model_repo_revision,
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    def _load_model(self, artifacts_path: Path) -> None:
        trust_remote_code = granite_vision_4_needs_remote_code(
            importlib.metadata.version("transformers")
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*torch_dtype.*deprecated.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*incorrect regex pattern.*",
                category=UserWarning,
            )
            self._processor = AutoProcessor.from_pretrained(
                artifacts_path,
                trust_remote_code=trust_remote_code,
            )
            self._model_max_length = self._processor.tokenizer.model_max_length
            self._model = AutoModelForImageTextToText.from_pretrained(
                artifacts_path,
                device_map=self.device,
                dtype=torch.bfloat16,
                # The native granite4_vision Q-Former rejects an explicit sdpa
                # request before transformers 5.13; the transformers default
                # selects sdpa where the model supports it.
                _attn_implementation=(
                    "flash_attention_2"
                    if self.device.startswith("cuda")
                    and self.accelerator_options.cuda_use_flash_attention2
                    else None
                ),
                trust_remote_code=trust_remote_code,
            )
        self._model.eval()

    def predict_tables(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[TableStructurePrediction]:
        predictions: list[TableStructurePrediction] = []

        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                existing = page.predictions.tablestructure or TableStructurePrediction()
                page.predictions.tablestructure = existing
                predictions.append(existing)
                continue

            with TimeRecorder(conv_res, "table_structure"):
                assert page.predictions.layout is not None
                assert page.size is not None

                table_prediction = TableStructurePrediction()
                page.predictions.tablestructure = table_prediction

                clusters = [
                    c
                    for c in page.predictions.layout.clusters
                    if c.label in (DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX)
                ]

                if not clusters or not self.enabled:
                    predictions.append(table_prediction)
                    continue

                # Crop one image per table cluster from the page image
                valid_pairs = []
                for cluster in clusters:
                    crop = page.get_image(scale=1.0, cropbox=cluster.bbox)
                    if crop is not None:
                        valid_pairs.append((cluster, crop))

                if not valid_pairs:
                    predictions.append(table_prediction)
                    continue

                valid_clusters, valid_images = zip(*valid_pairs)

                conversations = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "<tables_otsl>"},
                            ],
                        }
                    ]
                    for _ in valid_images
                ]

                texts = [
                    self._processor.apply_chat_template(
                        conv, tokenize=False, add_generation_prompt=True
                    )
                    for conv in conversations
                ]

                inputs = self._processor(
                    text=texts,
                    images=list(valid_images),
                    return_tensors="pt",
                    padding=True,
                    do_pad=True,
                ).to(self.device)

                output_ids = cast(Any, self._model).generate(
                    **inputs,
                    max_new_tokens=self._model_max_length,
                    use_cache=True,
                )

                # Decode only generated tokens (strip input prompt tokens)
                output_texts = [
                    self._processor.decode(
                        output_ids[i, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                    for i in range(len(valid_images))
                ]

                for cluster, raw_text in zip(valid_clusters, output_texts):
                    _log.debug(
                        f"GraniteVision table [{cluster.id}] raw output: {raw_text!r}"
                    )
                    try:
                        otsl_seq, table_cells, num_rows, num_cols = parse_otsl_output(
                            raw_text
                        )
                    except Exception as exc:
                        _log.warning(
                            f"Failed to parse OTSL output for table cluster {cluster.id}: {exc}"
                        )
                        otsl_seq, table_cells, num_rows, num_cols = [], [], 0, 0

                    # Apply the shared TableFormer-output cleanup helpers.
                    # `recover_leftover_words` is intentionally omitted —
                    # this model does not have a `tcells` input-token list
                    # to compare against the predicted grid.
                    table_cells = consolidate_duplicate_spans(table_cells)
                    promote_wrapped_header_rows(table_cells, num_rows)
                    table_cells, num_rows, num_cols, did_split = (
                        split_into_structural_subtables(
                            table_cells=table_cells,
                            table_cluster=cluster,
                            page_clusters=page.predictions.layout.clusters,
                            table_map=table_prediction.table_map,
                            page_no=page.page_no,
                        )
                    )
                    if did_split:
                        otsl_seq = []

                    tbl = Table(
                        otsl_seq=otsl_seq,
                        table_cells=table_cells,
                        num_rows=num_rows,
                        num_cols=num_cols,
                        id=cluster.id,
                        page_no=page.page_no,
                        cluster=cluster,
                        label=cluster.label,
                    )
                    table_prediction.table_map[cluster.id] = tbl

                predictions.append(table_prediction)

        return predictions
