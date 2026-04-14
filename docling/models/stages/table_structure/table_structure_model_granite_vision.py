import logging
import re
import warnings
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, cast

import torch
from docling_core.types.doc import DocItemLabel, TableCell
from transformers import AutoModelForImageTextToText, AutoProcessor

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
from docling.models.base_table_model import BaseTableStructureModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

# OTSL tokens that represent content-bearing cells (produce a TableCell)
_CONTENT_TOKENS = {"fcel", "ecel", "ched", "rhed", "srow"}
# OTSL tokens that are span-extensions (no separate TableCell, affect span of predecessor)
_SPAN_TOKENS = {"lcel", "ucel", "xcel"}

# Regex to extract (tag_name, inner_text) from VLM OTSL output.
# Matches: <tag>text</tag>  OR  <tag/>  OR  <tag>  (self-closing / bare nl)
_TAG_RE = re.compile(
    r"<(?P<tag>[a-z]+)>(?P<text>.*?)</(?P=tag)>"  # <tag>text</tag>
    r"|<(?P<stag>[a-z]+)\s*/>"                     # <tag/>
    r"|<(?P<btag>[a-z]+)>",                        # <tag> (bare, e.g. <nl>, <lcel>)
    re.DOTALL,
)


def _parse_otsl_output(
    text: str,
) -> tuple[list[str], list[TableCell], int, int]:
    """Parse VLM OTSL text output into structured table data.

    Parameters
    ----------
    text:
        Raw VLM output string, e.g.
        ``"<ched>Name</ched><ched>Val</ched><nl><fcel>Foo</fcel><fcel>42</fcel><nl>"``

    Returns
    -------
    tuple of (otsl_seq, table_cells, num_rows, num_cols)
        otsl_seq: list of bare tag names, e.g. ["ched", "ched", "nl", "fcel", "fcel", "nl"]
        table_cells: list of TableCell (bbox always None)
        num_rows: int
        num_cols: int
    """
    if not text or not text.strip():
        return [], [], 0, 0

    # Extract (tag, inner_text) pairs
    token_pairs: list[tuple[str, str]] = []
    for m in _TAG_RE.finditer(text):
        if m.group("tag"):
            token_pairs.append((m.group("tag"), m.group("text") or ""))
        elif m.group("stag"):
            token_pairs.append((m.group("stag"), ""))
        elif m.group("btag"):
            token_pairs.append((m.group("btag"), ""))

    if not token_pairs:
        return [], [], 0, 0

    otsl_seq = [tag for tag, _ in token_pairs]

    # Split into rows on "nl" tokens
    rows: list[list[tuple[str, str]]] = [
        list(group)
        for k, group in groupby(token_pairs, lambda x: x[0] == "nl")
        if not k
    ]

    if not rows:
        return otsl_seq, [], 0, 0

    num_rows = len(rows)
    num_cols = max(len(row) for row in rows)

    # Pad rows to equal width
    grid: list[list[tuple[str, str]]] = [
        row + [("", "")] * (num_cols - len(row)) for row in rows
    ]

    table_cells: list[TableCell] = []
    for row_idx, row in enumerate(grid):
        for col_idx, (tag, inner_text) in enumerate(row):
            if tag not in _CONTENT_TOKENS:
                continue

            # Detect colspan: count consecutive span-extension tokens to the right
            colspan = 1
            for c in range(col_idx + 1, num_cols):
                if grid[row_idx][c][0] in _SPAN_TOKENS:
                    colspan += 1
                else:
                    break

            # Detect rowspan: count consecutive span-extension tokens below
            rowspan = 1
            for r in range(row_idx + 1, num_rows):
                if grid[r][col_idx][0] in _SPAN_TOKENS:
                    rowspan += 1
                else:
                    break

            cell = TableCell(
                text=inner_text,
                bbox=None,
                row_span=rowspan,
                col_span=colspan,
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + rowspan,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + colspan,
                column_header=(tag == "ched"),
                row_header=(tag == "rhed"),
                row_section=(tag == "srow"),
            )
            table_cells.append(cell)

    return otsl_seq, table_cells, num_rows, num_cols


class GraniteVisionTableStructureModel(BaseTableStructureModel):
    """Table structure model using ibm-granite/granite-4.0-3b-vision with <tables_otsl>."""

    _model_repo_id: ClassVar[str] = "ibm-granite/granite-4.0-3b-vision"
    _model_repo_folder: ClassVar[str] = "ibm-granite--granite-4.0-3b-vision"
    _model_repo_revision: ClassVar[str] = "f0d034897bae1cd438c961c8c170a3a3089ebf01"

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: GraniteVisionTableStructureOptions,
        accelerator_options: AcceleratorOptions,
        enable_remote_services: Literal[False] = False,
    ):
        self.enabled = enabled
        self.options = options

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
        local_dir: Optional[Path] = None,
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
                trust_remote_code=True,
            )
            self._model_max_length = self._processor.tokenizer.model_max_length
            self._model = AutoModelForImageTextToText.from_pretrained(
                artifacts_path,
                device_map=self.device,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        cast(Any, self._model).merge_lora_adapters()
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
                        output_ids[i, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    for i in range(len(valid_images))
                ]

                for cluster, raw_text in zip(valid_clusters, output_texts):
                    _log.debug(
                        f"GraniteVision table [{cluster.id}] raw output: {raw_text!r}"
                    )
                    try:
                        otsl_seq, table_cells, num_rows, num_cols = _parse_otsl_output(
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
