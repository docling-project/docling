import logging
import re
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureClassificationMetaField,
    PictureItem,
    PictureMeta,
    TableCell,
    TableData,
    TabularChartMetaField,
)
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device

_log = logging.getLogger(__name__)


class ChartExtractionModelOptions(BaseModel):
    kind: Literal["chart_extraction"] = "chart_extraction"


class ChartExtractionModelGraniteVision(BaseItemAndImageEnrichmentModel):
    SUPPORTED_CHART_TYPES = ["bar_chart", "pie_chart", "line_chart"]

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: ChartExtractionModelOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """
        Initializes the CodeFormulaModel with the given configuration.

        Parameters
        ----------
        enabled : bool
            True if the model is enabled, False otherwise.
        artifacts_path : Path
            Path to the directory containing the model artifacts.
        options : CodeFormulaModelOptions
            Configuration options for the model.
        accelerator_options : AcceleratorOptions
            Options specifying the device and number of threads for acceleration.
        """
        self.enabled = enabled
        self.options = options

        if self.enabled:
            self.device = decide_device(
                accelerator_options.device,
                supported_devices=[AcceleratorDevice.CPU, AcceleratorDevice.CUDA],
            )

            if artifacts_path is None:
                artifacts_path = self.download_models()

            self._processor = AutoProcessor.from_pretrained(
                artifacts_path,
            )
            self._model_max_length = self._processor.tokenizer.model_max_length
            self._model = AutoModelForImageTextToText.from_pretrained(
                artifacts_path, device_map=self.device
            )
            self._model.eval()

    @staticmethod
    def download_models(
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        return download_hf_model(
            repo_id="ibm-granite/granite-vision-3.3-2b-chart2csv-preview",
            # Let's pin it to a specific commit to reduce potential regression errors
            revision="6e1fbaae4604ecc85f4f371416d82154ca49ad67", 
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        """
        Determines if a given element in a document can be processed by the model.

        Parameters
        ----------
        doc : DoclingDocument
            The document being processed.
        element : NodeItem
            The element within the document to check.

        Returns
        -------
        bool
            True if the element can be processed, False otherwise.
        """
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

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        """
        Processes the given batch of elements and enriches them with predictions.

        Parameters
        ----------
        doc : DoclingDocument
            The document being processed.
        element_batch : Iterable[ItemAndImageEnrichmentElement]
            A batch of elements to be processed.

        Returns
        -------
        Iterable[Any]
            An iterable of enriched elements.
        """
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

        # autoregressively complete prompt for batch
        output_ids = self._model.generate(
            **inputs, max_new_tokens=self._model_max_length
        )

        output_texts = self._processor.batch_decode(
            output_ids, skip_special_tokens=False
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
            # Post-process to extract DataFrame
            try:
                dataframe = self._extract_csv_to_dataframe(text)

                # In convert_batch_images, after extracting DataFrame:
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

        Args:
            decoded_text: The decoded output from the model

        Returns:
            pandas DataFrame containing the CSV data
        """
        # Extract the assistant's response
        assistant_match = re.search(r"<\|assistant\|>\s*(.*)", decoded_text, re.DOTALL)
        if not assistant_match:
            raise ValueError("Could not find assistant response in decoded text")

        assistant_response = assistant_match.group(1).strip()

        # Remove code block markers - handle both ``` and ````
        # First remove outer code blocks
        csv_content = re.sub(r"^```+(?:csv)?\s*", "", assistant_response)
        csv_content = re.sub(r"```+\s*$", "", csv_content)

        # Clean up any remaining code block markers
        csv_content = csv_content.strip()
        csv_content = re.sub(r"^```+(?:csv)?\s*", "", csv_content)
        csv_content = re.sub(r"```+\s*$", "", csv_content)

        # Convert to DataFrame
        try:
            dataframe = pd.read_csv(StringIO(csv_content), header=None)
            return dataframe
        except Exception as e:
            _log.error(f"Error parsing CSV: {e}")
            _log.error(f"CSV content:\n{csv_content}")
            raise

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

        Args:
            df: The pandas DataFrame to convert

        Returns:
            TableData object containing the table structure
        """
        table_cells = []

        # Infer if first row is header: check if all values in first row are non-numeric
        first_row_is_header = False
        if len(df) > 0:
            first_row = df.iloc[0]
            first_row_is_header = all(not self._is_numeric(val) for val in first_row)

        # Add header row cells if inferred
        if first_row_is_header:
            for col_idx, col_name in enumerate(df.columns):
                cell = TableCell(
                    text=str(col_name),
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

        # Add data cells
        row_offset = 1 if first_row_is_header else 0
        for row_idx, row in df.iterrows():
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
                    start_row_offset_idx=int(row_idx) + row_offset,
                    end_row_offset_idx=int(row_idx) + row_offset + 1,
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

        # Calculate total rows (header + data rows)
        num_rows = len(df) + (1 if first_row_is_header else 0)
        num_cols = len(df.columns)

        return TableData(table_cells=table_cells, num_rows=num_rows, num_cols=num_cols)
