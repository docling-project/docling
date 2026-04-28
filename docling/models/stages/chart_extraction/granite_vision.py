import logging
import re
import warnings
from abc import abstractmethod
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any, List, Literal, Optional, cast

import pandas as pd
import torch
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
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.chart_extraction_options import (
    ChartExtractionModelKind,
    ChartExtractionModelOptions,
)
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device

_log = logging.getLogger(__name__)


class _BaseChartExtractionModelGraniteVision(BaseItemAndImageEnrichmentModel):
    SUPPORTED_CHART_TYPES = ["bar_chart", "pie_chart", "line_chart"]
    _model_repo_folder: str
    _model_repo_id: str
    _model_repo_revision: str

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: ChartExtractionModelOptions,
        accelerator_options: AcceleratorOptions,
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
                    f"Model artifacts not found at {artifacts_path / self._model_repo_folder}, they will be downloaded."
                )

            self._load_model(artifacts_path)

    @abstractmethod
    def _load_model(self, artifacts_path: Path) -> None:
        """Load the processor and model from the given artifacts path."""

    @classmethod
    def download_models(
        cls,
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        return download_hf_model(
            repo_id=cls._model_repo_id,
            # Let's pin it to a specific commit to reduce potential regression errors
            revision=cls._model_repo_revision,
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

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
        return main_pred.class_name in self.SUPPORTED_CHART_TYPES

    def _is_numeric(self, value) -> bool:
        """Check if a value is numeric (int or float)."""
        if pd.isna(value):
            return False
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _dataframe_to_tabledata(self, df: pd.DataFrame) -> TableData:
        """
        Transform a pandas DataFrame into a TableData object.

        Automatically infers if the first row is a header by checking if all
        values in the first row are non-numeric.
        """
        table_cells = []

        # Infer if first row is header: check if all values in first row are non-numeric
        first_row_is_header = False
        if len(df) > 0:
            first_row = df.iloc[0]
            first_row_is_header = all(not self._is_numeric(val) for val in first_row)

        # Add header row cells if inferred
        if first_row_is_header:
            for col_idx, value in enumerate(df.iloc[0]):
                cell = TableCell(
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
                table_cells.append(cell)

        # Add data cells (skip the first row if it was used as header)
        data_df = df.iloc[1:] if first_row_is_header else df
        row_offset = 1 if first_row_is_header else 0
        for row_idx, (_idx, row) in enumerate(data_df.iterrows()):
            for col_idx, value in enumerate(row):
                # Convert value to string, handling NaN and None
                if pd.isna(value):
                    text = ""
                else:
                    text = str(value)

                # Check if the value is numeric - non-numeric cells are row headers
                is_row_header = not self._is_numeric(value)

                cell = TableCell(
                    text=text,
                    start_row_offset_idx=row_idx + row_offset,
                    end_row_offset_idx=row_idx + row_offset + 1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    row_span=1,
                    col_span=1,
                    column_header=False,
                    row_header=is_row_header,
                    row_section=False,
                    fillable=False,
                )
                table_cells.append(cell)

        # Total rows equals DataFrame length in both cases:
        # with header: 1 header + (len(df) - 1) data rows = len(df)
        # without header: len(df) data rows
        num_rows = len(df)
        num_cols = len(df.columns)

        return TableData(table_cells=table_cells, num_rows=num_rows, num_cols=num_cols)


class ChartExtractionModelGraniteVision(_BaseChartExtractionModelGraniteVision):
    _model_repo_folder = "ibm-granite--granite-vision-3.3-2b-chart2csv-preview"
    _model_repo_id = "ibm-granite/granite-vision-3.3-2b-chart2csv-preview"
    _model_repo_revision = "6e1fbaae4604ecc85f4f371416d82154ca49ad67"

    def _load_model(self, artifacts_path: Path) -> None:
        self._processor = AutoProcessor.from_pretrained(
            artifacts_path,
            trust_remote_code=True,
        )
        self._model_max_length = self._processor.tokenizer.model_max_length
        self._model = AutoModelForImageTextToText.from_pretrained(
            artifacts_path,
            device_map=self.device,
            dtype="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        images: List[Image.Image] = []
        elements: List[PictureItem] = []
        for el in element_batch:
            elements.append(el.item)  # type: ignore[arg-type]
            images.append(el.image)

        # Create a batch of conversations
        conversations = []
        for image in images:
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},  # <-- PIL Image here
                            {
                                "type": "text",
                                "text": "Convert the information in this chart into a data table in CSV format.",
                            },
                        ],
                    },
                ]
            )

        # Process batch in a single call
        inputs = self._processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(self.device)

        eos_ids = [
            self._processor.tokenizer.eos_token_id,
            self._processor.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        ]

        # autoregressively complete prompt for batch
        output_ids = cast(Any, self._model).generate(
            **inputs,
            max_new_tokens=self._model_max_length,
            eos_token_id=eos_ids,
        )

        output_texts = self._processor.batch_decode(
            output_ids, skip_special_tokens=True
        )

        chart_data: list[Optional[TabularChartMetaField]] = self._post_process(
            outputs=output_texts
        )

        for item, tabular_chart in zip(elements, chart_data):
            if (tabular_chart is not None) and isinstance(item, PictureItem):
                if (item.meta is not None) and isinstance(item.meta, PictureMeta):
                    item.meta.tabular_chart = tabular_chart
                else:
                    meta = PictureMeta(tabular_chart=tabular_chart)
                    item.meta = meta

            yield item

    def _post_process(
        self, outputs: list[str]
    ) -> list[Optional[TabularChartMetaField]]:
        chart_data: list[Optional[TabularChartMetaField]] = []

        for i, text in enumerate(outputs):
            try:
                dataframe = self._extract_csv_to_dataframe(text)
                table_data = self._dataframe_to_tabledata(dataframe)
                chart_data.append(TabularChartMetaField(chart_data=table_data))
            except Exception as e:
                _log.error(f"Failed to extract DataFrame for image {i}: {e}")
                chart_data.append(None)

        return chart_data

    def _extract_csv_to_dataframe(self, decoded_text: str) -> pd.DataFrame:
        """
        Extract CSV content from decoded text and convert to DataFrame.

        Handles:
        - Chat format with <|assistant|> tags
        - Nested code blocks (```csv ``` inside ```)
        - Various CSV formatting issues
        """
        # Extract the assistant's response
        assistant_match = re.search(r"<\|assistant\|>\s*(.*)", decoded_text, re.DOTALL)
        if not assistant_match:
            raise ValueError("Could not find assistant response in decoded text")

        assistant_response = assistant_match.group(1).strip()

        # Extract the first CSV code block (```csv ... ```)
        csv_match = re.search(r"```csv\s*\n(.*?)\n```", assistant_response, re.DOTALL)
        if csv_match:
            csv_content = csv_match.group(1).strip()
        else:
            # Fallback: take content up to first <|end_of_text|> and strip code block markers
            csv_content = assistant_response.split("<|end_of_text|>")[0].strip()
            csv_content = re.sub(r"^```+(?:csv)?\s*", "", csv_content)
            csv_content = re.sub(r"```+\s*$", "", csv_content)
            csv_content = csv_content.strip()

        try:
            dataframe = pd.read_csv(StringIO(csv_content), header=None)
            return dataframe
        except Exception as e:
            _log.error(f"Error parsing CSV: {e}")
            _log.error(f"CSV content:\n{csv_content}")
            raise


class ChartExtractionModelGraniteVisionV4(_BaseChartExtractionModelGraniteVision):
    _model_repo_folder = "ibm-granite--granite-vision-4.1-4b"
    _model_repo_id = "ibm-granite/granite-vision-4.1-4b"
    _model_repo_revision = "dd48e97503de471803850df70843cf9eb5da8712"

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
        if hasattr(self._model, "merge_lora_adapters"):
            cast(Any, self._model).merge_lora_adapters()
        self._model.eval()

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        images: List[Image.Image] = []
        elements: List[PictureItem] = []
        for el in element_batch:
            elements.append(el.item)  # type: ignore[arg-type]
            images.append(el.image)

        active_prompts: list[str] = []
        if self.options.chart2csv:
            active_prompts.append("<chart2csv>")
        if self.options.chart2summary:
            active_prompts.append("<chart2summary>")
        if self.options.chart2code:
            active_prompts.append("<chart2code>")

        if not active_prompts:
            for item in elements:
                yield item
            return

        # Flat batch: each image paired with each active prompt
        batch_images = [img for img in images for _ in active_prompts]
        batch_prompts = [prompt for _ in images for prompt in active_prompts]

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for prompt in batch_prompts
        ]
        texts = [
            self._processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            for conv in conversations
        ]
        inputs = self._processor(
            text=texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            do_pad=True,
        ).to(self.device)

        output_ids = cast(Any, self._model).generate(
            **inputs,
            max_new_tokens=self._model_max_length,
            use_cache=True,
        )

        # Decode only the generated tokens, not the input prompt
        output_texts = [
            self._processor.decode(
                output_ids[i, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            for i in range(len(batch_prompts))
        ]

        # Apply results to each element's PictureMeta
        n_prompts = len(active_prompts)
        for img_idx, item in enumerate(elements):
            if not isinstance(item, PictureItem):
                yield item
                continue

            if item.meta is None or not isinstance(item.meta, PictureMeta):
                item.meta = PictureMeta()

            for prompt_idx, prompt in enumerate(active_prompts):
                result = output_texts[img_idx * n_prompts + prompt_idx]
                _log.debug(f"chart extraction [{prompt}] image {img_idx}: {result}")
                try:
                    if prompt == "<chart2csv>":
                        chart_df = self._extract_csv_to_dataframe(result)
                        item.meta.tabular_chart = TabularChartMetaField(
                            chart_data=self._dataframe_to_tabledata(chart_df)
                        )
                    elif prompt == "<chart2summary>":
                        item.meta.description = DescriptionMetaField(text=result)
                    elif prompt == "<chart2code>":
                        code = self._extract_python_code(result)
                        if code is not None:
                            item.meta.code = CodeMetaField(
                                text=code, language=CodeLanguageLabel.PYTHON
                            )
                except Exception as e:
                    _log.error(f"Failed to process [{prompt}] for image {img_idx}: {e}")

            yield item

    def _extract_csv_to_dataframe(self, decoded_text: str) -> pd.DataFrame:
        # decoded_text is already the raw generated output (no conversation wrapper)
        csv_match = re.search(r"```csv\s*\n(.*?)\n```", decoded_text, re.DOTALL)
        if not csv_match:
            raise ValueError("No ```csv``` block found in model output")
        csv_content = csv_match.group(1).strip()
        try:
            return pd.read_csv(StringIO(csv_content), header=None)
        except Exception as e:
            _log.error(f"Error parsing CSV: {e}")
            _log.error(f"CSV content:\n{csv_content}")
            raise

    def _extract_python_code(self, decoded_text: str) -> Optional[str]:
        python_match = re.search(r"```python\s*\n(.*?)\n```", decoded_text, re.DOTALL)
        if not python_match:
            return None
        return python_match.group(1).strip()
