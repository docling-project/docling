# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import importlib.metadata
import logging
import re
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple, cast

import torch
from docling_core.types.doc import DocItemLabel, TableCell
from docling_core.types.doc.page import TextCell
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    StoppingCriteriaList,
)

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import (
    Cluster,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    Page,
    Table,
    TableStructurePrediction,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
from docling.models.base_table_model import (
    BaseTableStructureModel,
    keep_table_text_as_child,
    page_cells_in_table,
)
from docling.models.utils.generation_utils import TailRepetitionStopper
from docling.models.utils.hf_model_download import download_hf_model
from docling.models.utils.hf_stopping_criteria import HFStoppingCriteriaWrapper
from docling.utils.accelerator_utils import decide_device
from docling.utils.granite_vision_utils import granite_vision_4_needs_remote_code
from docling.utils.otsl import parse_otsl_output
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

_ROW_BREAK = "<nl>"
_WORD = re.compile(r"\w+")
# Fewer native words than this give a share too coarse to judge a transcription.
_MIN_WORDS_FOR_TEXT_CHECK = 4


class TextRecall(NamedTuple):
    """How many of the words in the PDF text under a table the model transcribed."""

    found: int
    total: int

    @property
    def share(self) -> float:
        return self.found / self.total


def native_text_recall(
    native_cells: Sequence[TextCell], table_cells: Sequence[TableCell]
) -> TextRecall | None:
    """Compare the transcribed cells with the PDF text layer under the table.

    The Granite stage transcribes cell text from the image and never reads the
    page's text. When the page has a native text layer, the text lines under
    the table box say what the cells must contain, so a transcription that
    lost most of their words is a failed structure even when the generation
    ended normally. The caller passes native lines only: OCR cells are a
    second reading of the same pixels, not ground truth. Words are compared
    as case-folded runs of letters and digits, so spacing and punctuation
    differences between the two readings do not count.

    Returns None when there is no native text to compare with.
    """
    native = Counter(
        word for cell in native_cells for word in _WORD.findall(cell.text.casefold())
    )
    total = sum(native.values())
    if total < _MIN_WORDS_FOR_TEXT_CHECK:
        return None
    transcribed = Counter(
        word for cell in table_cells for word in _WORD.findall(cell.text.casefold())
    )
    found = sum(min(count, transcribed[word]) for word, count in native.items())
    return TextRecall(found=found, total=total)


@dataclass(frozen=True)
class _GeneratedOtsl:
    """One table's decoded OTSL and how its generation ended."""

    text: str
    hit_budget: bool
    dropped_chars: int


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
        # The window it inspects must hold that span. The OTSL tags are three
        # to four tokens each in this tokenizer, about 1.7 characters per
        # token on a row of numbers, so 1024 tokens show ~1,800 characters:
        # too few for 32 copies of any row wider than ~55 characters, and the
        # loop then ran to the budget. 4096 tokens cover units up to ~220
        # characters at about 2 ms per generation step.
        self._repetition_stopper: TailRepetitionStopper | None = (
            TailRepetitionStopper(
                min_repeats=32, min_span=640, max_unit=256, lookback_tokens=4096
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

    def _decode_generated(
        self, output_ids: Any, prompt_len: int, row: int
    ) -> _GeneratedOtsl:
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
        text = self._processor.decode(generated, skip_special_tokens=True)
        dropped_chars = 0
        if self._repetition_stopper is not None:
            text, dropped_chars = self._drop_repeated_rows(text)
        return _GeneratedOtsl(
            text=text,
            hit_budget=generated_count >= self.options.max_new_tokens,
            dropped_chars=dropped_chars,
        )

    def _drop_repeated_rows(self, text: str) -> tuple[str, int]:
        """Remove a looping OTSL tail without damaging the last real row.

        The generic ``TailRepetitionStopper.strip`` walks the periodic run back
        character by character, which on tag-structured text also eats the
        closing tags shared with the preceding real row. Whole copies of the
        repeated unit are peeled off instead, and a partial row left by a
        stop in mid-unit is cut at the last row break. The loop usually starts
        on a real row that the model then repeats, so one copy of a unit that
        spans whole rows is put back.

        Returns the cleaned text and the number of characters removed.
        """
        assert self._repetition_stopper is not None
        unit = self._repetition_stopper.repeated_unit(text)
        if unit is None:
            return text, 0
        unit_text = text[-unit:]
        kept = text
        while kept.endswith(unit_text):
            kept = kept[: -len(unit_text)]
        last_row_break = kept.rfind(_ROW_BREAK)
        kept = kept[: last_row_break + len(_ROW_BREAK)] if last_row_break >= 0 else ""
        # The unit was sliced from the end of the text, so it may start in the
        # middle of a row; rotate it to begin at a row start before keeping it.
        row_break = unit_text.rfind(_ROW_BREAK)
        if row_break >= 0:
            row_end = row_break + len(_ROW_BREAK)
            kept += unit_text[row_end:] + unit_text[:row_end]
        return kept, len(text) - len(kept)

    def _table_from_output(
        self,
        conv_res: ConversionResult,
        page: Page,
        cluster: Cluster,
        generated: _GeneratedOtsl,
    ) -> Table:
        """Parse one table's OTSL, check it, and fall back to the page text if it failed.

        A structure counts as failed when the generation was cut off by the
        token budget, when the output holds no table, or when the page's text
        layer shows that the transcription lost most of the words under the
        table. A failed structure is discarded and the text under the table
        is kept as a text child, so nothing the PDF says is lost; when there
        is no such text (a scanned page without OCR) a partial structure is
        kept rather than nothing. Every failure, and every trimmed loop that
        the text layer could not confirm, is recorded in ``conv_res.errors``
        so the conversion reports partial success instead of silent loss.
        """
        problems: list[str] = []
        failed = False
        if generated.hit_budget:
            problems.append(
                f"the output hit the max_new_tokens limit ({self.options.max_new_tokens})"
            )
            failed = True
        if generated.dropped_chars:
            problems.append(
                "the output repeated the same fragment and "
                f"{generated.dropped_chars} characters of repeated rows were dropped"
            )
        try:
            otsl_seq, table_cells, num_rows, num_cols = parse_otsl_output(
                generated.text
            )
        except Exception as exc:
            problems.append(f"the OTSL output could not be parsed ({exc})")
            otsl_seq, table_cells, num_rows, num_cols = [], [], 0, 0
            failed = True
        if not failed and num_rows == 0:
            problems.append("the output holds no table structure")
            failed = True

        recall: TextRecall | None = None
        if not failed and self.options.min_text_recall > 0:
            native_cells = [
                cell for cell in page_cells_in_table(cluster, page) if not cell.from_ocr
            ]
            recall = native_text_recall(native_cells, table_cells)
            if recall is not None and recall.share < self.options.min_text_recall:
                problems.append(
                    f"the transcribed cells hold {recall.found} of the "
                    f"{recall.total} words in the PDF text under the table "
                    f"(min_text_recall={self.options.min_text_recall})"
                )
                failed = True

        if failed:
            keep_table_text_as_child(cluster, page)
            if cluster.children:
                otsl_seq, table_cells, num_rows, num_cols = [], [], 0, 0
                problems.append(
                    "the structure was discarded and the text under the table "
                    "was kept as plain text"
                )
            elif num_rows:
                problems.append(
                    "no text was found under the table, so the partial "
                    "structure was kept"
                )
            else:
                problems.append("no text was found under the table")

        if problems:
            confirmed = not failed and recall is not None
            if not failed and not confirmed:
                problems.append(
                    "the kept rows could not be checked against the PDF text "
                    "under the table"
                )
            message = (
                f"GraniteVision table {cluster.id} on page {page.page_no}: "
                + "; ".join(problems)
                + "."
            )
            if confirmed:
                _log.warning(
                    "%s The kept rows match the PDF text under the table.", message
                )
            else:
                _log.warning(message)
                conv_res.errors.append(
                    ErrorItem(
                        component_type=DoclingComponentType.MODEL,
                        module_name=type(self).__name__,
                        error_message=message,
                        category=FailureCategory.INFERENCE_FAILURE,
                        page_no=page.page_no,
                    )
                )

        return Table(
            otsl_seq=otsl_seq,
            table_cells=table_cells,
            num_rows=num_rows,
            num_cols=num_cols,
            id=cluster.id,
            page_no=page.page_no,
            cluster=cluster,
            label=cluster.label,
        )

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
                    crop = page.get_image(
                        scale=self.options.image_scale, cropbox=cluster.bbox
                    )
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
                outputs = [
                    self._decode_generated(output_ids, prompt_len, i)
                    for i in range(len(valid_images))
                ]

                for cluster, generated in zip(valid_clusters, outputs):
                    _log.debug(
                        f"GraniteVision table [{cluster.id}] raw output: {generated.text!r}"
                    )
                    table_prediction.table_map[cluster.id] = self._table_from_output(
                        conv_res, page, cluster, generated
                    )

            predictions.append(table_prediction)

        return predictions
