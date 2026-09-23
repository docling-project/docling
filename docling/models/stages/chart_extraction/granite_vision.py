# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import re
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
from docling_core.types.doc import (
    CodeLanguageLabel,
    DescriptionMetaField,
    DoclingDocument,
    NodeItem,
    PictureClassificationMetaField,
    PictureItem,
    PictureMeta,
    TableCell,
    TableData,
    TabularChartMetaField,
)
from docling_core.types.doc.document import CodeMetaField

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.chart_extraction_options import (
    ChartExtractionOutputFormat,
    ChartExtractionVlmEngineOptions,
)
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.models.inference_engines.vlm import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineType,
    create_vlm_engine,
)

_log = logging.getLogger(__name__)

SUPPORTED_CHART_TYPES = ["bar_chart", "pie_chart", "line_chart"]

# Natural-language equivalents for the special-token prompts used by the
# fine-tuned Granite Vision model.  These are only substituted when the user
# explicitly sets ``use_natural_language_prompts=True`` on the options (e.g.
# when serving a GGUF quantization that lacks the fine-tuned special tokens).
# API engines that serve the HF fine-tune via vLLM support the special tokens
# natively and should NOT use this map.
_NL_PROMPT_MAP: dict[str, str] = {
    "<chart2csv>": (
        "Convert the information in this chart into a data table in CSV format "
        "with a header row and numeric values."
    ),
    "<chart2summary>": "Describe this chart in a few sentences.",
    "<chart2code>": (
        "Write Python code using matplotlib that recreates this chart. "
        "Return only a fenced ```python code block."
    ),
}


class ChartExtractionVlmEngineModel(BaseItemAndImageEnrichmentModel):
    """Chart extraction stage using the unified VLM engine system.

    Supports all engine types (Transformers, MLX, API, vLLM) through the
    engine factory. Uses :class:`ChartExtractionVlmEngineOptions` which has
    a preset mechanism mirroring picture description and code/formula stages.

    Example::

        from docling.datamodel.chart_extraction_options import ChartExtractionVlmEngineOptions

        opts = ChartExtractionVlmEngineOptions.from_preset("granite_vision_v4")
        model = ChartExtractionVlmEngineModel(
            enabled=True,
            artifacts_path=None,
            options=opts,
            accelerator_options=AcceleratorOptions(),
            enable_remote_services=False,
        )
    """

    images_scale: float = 2.0

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: ChartExtractionVlmEngineOptions,
        accelerator_options: AcceleratorOptions,
        enable_remote_services: bool = False,
    ) -> None:
        self.enabled = enabled
        self.options = options
        self.engine: Optional[BaseVlmEngine] = None

        if self.enabled:
            self.engine = create_vlm_engine(
                options=options.engine_options,
                model_spec=options.model_spec,
                accelerator_options=accelerator_options,
                artifacts_path=artifacts_path,
                enable_remote_services=enable_remote_services,
            )

    # ------------------------------------------------------------------
    # BaseItemAndImageEnrichmentModel protocol
    # ------------------------------------------------------------------

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        if not self.enabled:
            return False
        if not isinstance(element, PictureItem):
            return False
        if element.meta is None or not isinstance(element.meta, PictureMeta):
            return False
        if element.meta.classification is None or not isinstance(
            element.meta.classification, PictureClassificationMetaField
        ):
            return False
        main_pred = element.meta.classification.get_main_prediction()
        return main_pred.class_name in SUPPORTED_CHART_TYPES

    def _resolve_runtime_engine_type(self) -> VlmEngineType:
        selected_engine_type = getattr(self.engine, "selected_engine_type", None)
        if selected_engine_type is not None:
            return selected_engine_type
        return self.options.engine_options.engine_type

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            for el in element_batch:
                yield el.item
            return

        elements: List[PictureItem] = []
        images = []
        for el in element_batch:
            elements.append(el.item)  # type: ignore[arg-type]
            images.append(el.image)

        # The prompts that are active for each image (one pass per prompt per image).
        active_prompts = self.options.active_prompts()
        if not active_prompts:
            for item in elements:
                yield item
            return

        # Translate special-token prompts to natural-language only when the user
        # has explicitly requested it (e.g. for a GGUF deployment that lacks the
        # fine-tuned tokens). HF fine-tunes served via vLLM/OpenAI-compat support
        # the special tokens natively — do not substitute by default.
        use_nl = self.options.use_natural_language_prompts

        # Forward the full generation configuration from model_spec so that
        # stop strings, temperature, and extra flags reach the engine.
        model_spec = self.options.model_spec
        engine_type = self._resolve_runtime_engine_type()
        stop_strings = list(model_spec.stop_strings)
        extra_generation_config = model_spec.get_runtime_input_extra_config(engine_type)

        # Build a flat batch: image x prompt, keeping them in sync
        batch_inputs: list[VlmEngineInput] = []
        for image in images:
            for prompt in active_prompts:
                wire_prompt = _NL_PROMPT_MAP.get(prompt, prompt) if use_nl else prompt
                batch_inputs.append(
                    VlmEngineInput(
                        image=image,
                        prompt=wire_prompt,
                        temperature=model_spec.temperature,
                        max_new_tokens=model_spec.max_new_tokens,
                        stop_strings=stop_strings,
                        extra_generation_config=extra_generation_config,
                    )
                )

        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        outputs = list(self.engine.predict_batch(batch_inputs))

        n_prompts = len(active_prompts)
        for img_idx, item in enumerate(elements):
            if not isinstance(item, PictureItem):
                yield item
                continue

            if item.meta is None or not isinstance(item.meta, PictureMeta):
                item.meta = PictureMeta()

            handler = _OUTPUT_FORMAT_HANDLERS.get(self.options.output_format)
            if handler is None:
                _log.error(
                    f"No handler registered for output_format "
                    f"{self.options.output_format!r}; skipping image {img_idx}."
                )
                yield item
                continue

            for prompt_idx, prompt in enumerate(active_prompts):
                result = outputs[img_idx * n_prompts + prompt_idx].text
                _log.debug(
                    f"chart extraction [{prompt}] image {img_idx}: {result[:120]}"
                )
                try:
                    handler(prompt, result, item)
                except Exception as exc:
                    _log.error(
                        f"Failed to process [{prompt}] for image {img_idx}: {exc}"
                    )

            yield item

    def __del__(self) -> None:
        if self.engine is not None:
            try:
                self.engine.cleanup()
            except Exception as exc:
                _log.warning(f"Error cleaning up chart extraction engine: {exc}")


# ---------------------------------------------------------------------------
# Output format handlers
# ---------------------------------------------------------------------------
# Each handler has the signature:
#   (prompt: str, result: str, item: PictureItem) -> None
# and is responsible for interpreting `result` for a single (prompt, image)
# pair and writing the parsed value into `item.meta`.
#
# Register a new callable in _OUTPUT_FORMAT_HANDLERS to support a model whose
# output shape differs from the existing ones.

_ChartOutputHandler = Callable[[str, str, PictureItem], None]


def _handle_granite_vision_charts(prompt: str, result: str, item: PictureItem) -> None:
    """Parser for the Granite Vision chart model multi-pass protocol.

    * ``<chart2csv>``     → fenced ```csv``` block (or bare CSV)
    * ``<chart2summary>`` → plain text passthrough
    * ``<chart2code>``    → fenced ```python``` block
    """
    assert item.meta is not None  # guaranteed by the caller
    if prompt == "<chart2csv>":
        chart_df = _extract_csv_to_dataframe(result)
        item.meta.tabular_chart = TabularChartMetaField(
            chart_data=_dataframe_to_tabledata(chart_df)
        )
    elif prompt == "<chart2summary>":
        item.meta.description = DescriptionMetaField(text=result)
    elif prompt == "<chart2code>":
        code = _extract_python_code(result)
        if code is not None:
            item.meta.code = CodeMetaField(text=code, language=CodeLanguageLabel.PYTHON)
    else:
        _log.warning(
            f"Unknown prompt token {prompt!r} for output_format "
            f"{ChartExtractionOutputFormat.GRANITE_VISION_CHARTS!r}; skipping."
        )


_OUTPUT_FORMAT_HANDLERS: Dict[ChartExtractionOutputFormat, _ChartOutputHandler] = {
    ChartExtractionOutputFormat.GRANITE_VISION_CHARTS: _handle_granite_vision_charts,
}


# ---------------------------------------------------------------------------
# Shared post-processing helpers
# ---------------------------------------------------------------------------


def _is_numeric(value: object) -> bool:
    if pd.isna(value):  # type: ignore[arg-type]
        return False
    try:
        float(value)  # type: ignore[arg-type]
        return True
    except (ValueError, TypeError):
        return False


def _dataframe_to_tabledata(df: pd.DataFrame) -> TableData:
    """Convert a pandas DataFrame into a ``TableData`` object."""
    table_cells: list[TableCell] = []

    # Chart series labels can be numbers (for example, years). A text label in
    # the first column distinguishes a mixed header from an all-numeric data row.
    first_row_is_header = len(df) > 0 and (
        all(not _is_numeric(val) for val in df.iloc[0])
        or (len(df.columns) > 1 and not _is_numeric(df.iloc[0, 0]))
    )

    if first_row_is_header:
        for col_idx, value in enumerate(df.iloc[0]):
            table_cells.append(
                TableCell(
                    text=str(value),
                    start_row_offset_idx=0,
                    end_row_offset_idx=1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    row_span=1,
                    col_span=1,
                    column_header=True,
                    row_header=False,
                    row_section=False,
                    fillable=False,
                )
            )

    data_df = df.iloc[1:] if first_row_is_header else df
    row_offset = 1 if first_row_is_header else 0
    for row_idx, (_idx, row) in enumerate(data_df.iterrows()):
        for col_idx, value in enumerate(row):
            text = "" if pd.isna(value) else str(value)
            table_cells.append(
                TableCell(
                    text=text,
                    start_row_offset_idx=row_idx + row_offset,
                    end_row_offset_idx=row_idx + row_offset + 1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    row_span=1,
                    col_span=1,
                    column_header=False,
                    row_header=col_idx == 0 and not _is_numeric(value),
                    row_section=False,
                    fillable=False,
                )
            )

    return TableData(
        table_cells=table_cells, num_rows=len(df), num_cols=len(df.columns)
    )


def _extract_csv_to_dataframe(decoded_text: str) -> pd.DataFrame:
    """Parse CSV from the V4 model output (raw generated tokens, no wrapper)."""
    csv_match = re.search(r"```csv\s*\n(.*?)\n```", decoded_text, re.DOTALL)
    if csv_match:
        csv_content = csv_match.group(1).strip()
    else:
        csv_content = re.sub(r"^```+(?:csv)?\s*", "", decoded_text.strip())
        csv_content = re.sub(r"```+\s*$", "", csv_content).strip()
    try:
        return pd.read_csv(StringIO(csv_content), header=None)
    except Exception as exc:
        _log.error(f"Error parsing CSV: {exc}\nCSV content:\n{csv_content}")
        raise


def _extract_python_code(decoded_text: str) -> Optional[str]:
    python_match = re.search(r"```python\s*\n(.*?)\n```", decoded_text, re.DOTALL)
    if not python_match:
        return None
    return python_match.group(1).strip()
