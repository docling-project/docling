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
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    StoppingCriteriaList,
)

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
from docling.models.base_table_model import BaseTableStructureModel
from docling.models.utils.generation_utils import TailRepetitionStopper
from docling.models.utils.hf_model_download import download_hf_model
from docling.models.utils.hf_stopping_criteria import HFStoppingCriteriaWrapper
from docling.utils.accelerator_utils import decide_device
from docling.utils.granite_vision_utils import granite_vision_4_needs_remote_code
from docling.utils.otsl import parse_otsl_output
from docling.utils.profiling import TimeRecorder

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
        # OTSL legitimately repeats short fragments (runs of <ecel>, identical
        # rows), so the loop detector needs a longer repeated span than the
        # code/formula default before it trips. Units are up to one wide row.
        self._repetition_stopper: TailRepetitionStopper | None = (
            TailRepetitionStopper(
                min_repeats=32, min_span=640, max_unit=256, lookback_tokens=1024
            )
            if options.stop_on_repetition
            else None
        )

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

    def _generate(self, inputs: Any) -> Any:
        """Run bounded OTSL generation for a batch of table crops.

        The tokenizer's ``model_max_length`` is the transformers placeholder
        (1e30) for this model, so it must not serve as the token budget: a
        crop that never emits end-of-text would otherwise generate until the
        process is killed (issue #4657).
        """
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.options.max_new_tokens,
            "use_cache": True,
        }
        if self._repetition_stopper is not None:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [
                    HFStoppingCriteriaWrapper(
                        self._processor.tokenizer,
                        self._repetition_stopper,
                        skip_special_tokens=True,
                    )
                ]
            )
        return cast(Any, self._model).generate(**inputs, **gen_kwargs)

    def _decode_generated(self, output_ids: Any, prompt_len: int, row: int) -> str:
        """Decode one row of generated tokens and clean up a runaway tail."""
        generated = output_ids[row, prompt_len:]
        # Rows that finished early are padded to the longest row of the batch,
        # so count the row's own tokens before comparing with the budget.
        pad_token_id = self._processor.tokenizer.pad_token_id
        generated_count = (
            int((generated != pad_token_id).sum())
            if pad_token_id is not None
            else int(generated.shape[0])
        )
        if generated_count >= self.options.max_new_tokens:
            _log.warning(
                "GraniteVision table output hit the max_new_tokens limit (%d); "
                "the table may be truncated.",
                self.options.max_new_tokens,
            )
        text = self._processor.decode(generated, skip_special_tokens=True)
        if self._repetition_stopper is not None:
            text = self._drop_repeated_rows(text)
        return text

    def _drop_repeated_rows(self, text: str) -> str:
        """Remove a looping OTSL tail without damaging the last real row.

        The generic ``TailRepetitionStopper.strip`` walks the periodic run back
        character by character, which on tag-structured text also eats the
        closing tags shared with the preceding real row. Whole copies of the
        repeated unit are peeled off instead, and a partial row left by a
        stop in mid-unit is cut at the last row break.
        """
        assert self._repetition_stopper is not None
        unit = self._repetition_stopper.repeated_unit(text)
        if unit is None:
            return text
        unit_text = text[-unit:]
        kept = text
        while kept.endswith(unit_text):
            kept = kept[: -len(unit_text)]
        last_row_break = kept.rfind("<nl>")
        if last_row_break >= 0:
            kept = kept[: last_row_break + len("<nl>")]
        _log.warning(
            "GraniteVision table output repeated the same fragment; "
            "dropped %d characters of repeated rows.",
            len(text) - len(kept),
        )
        return kept

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

                output_ids = self._generate(inputs)

                # Decode only generated tokens (strip input prompt tokens)
                prompt_len = inputs["input_ids"].shape[1]
                output_texts = [
                    self._decode_generated(output_ids, prompt_len, i)
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
