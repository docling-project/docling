import re
from collections.abc import Iterable
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from docling_core.types.doc import (
    CodeItem,
    DocItemLabel,
    DoclingDocument,
    NodeItem,
    TextItem,
)
from docling_core.types.doc.labels import CodeLanguageLabel
from PIL import Image
from pydantic import BaseModel

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device


class CodeFormulaModelOptions(BaseModel):
    """
    Configuration options for the CodeFormulaModel.

    Attributes
    ----------
    kind : str
        Type of the model. Fixed value "code_formula".
    do_code_enrichment : bool
        True if code enrichment is enabled, False otherwise.
    do_formula_enrichment : bool
        True if formula enrichment is enabled, False otherwise.
    """

    kind: Literal["code_formula"] = "code_formula"
    do_code_enrichment: bool = True
    do_formula_enrichment: bool = True


class CodeFormulaModel(BaseItemAndImageEnrichmentModel):
    """
    Model for processing and enriching documents with code and formula predictions.

    Attributes
    ----------
    enabled : bool
        True if the model is enabled, False otherwise.
    options : CodeFormulaModelOptions
        Configuration options for the CodeFormulaModel.
    code_formula_model : CodeFormulaPredictor
        The predictor model for code and formula processing.

    Methods
    -------
    __init__(self, enabled, artifacts_path, accelerator_options, code_formula_options)
        Initializes the CodeFormulaModel with the given configuration options.
    is_processable(self, doc, element)
        Determines if a given element in a document can be processed by the model.
    __call__(self, doc, element_batch)
        Processes the given batch of elements and enriches them with predictions.
    """

    _model_repo_folder = "docling-project--CodeFormulaV2"
    elements_batch_size = 5
    images_scale = 1.67  # = 120 dpi, aligned with training data resolution
    expansion_factor = 0.18

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: CodeFormulaModelOptions,
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
                supported_devices=[
                    AcceleratorDevice.CPU,
                    AcceleratorDevice.CUDA,
                    AcceleratorDevice.XPU,
                ],
            )

            if artifacts_path is None:
                artifacts_path = self.download_models()
            else:
                artifacts_path = artifacts_path / self._model_repo_folder

            from transformers import AutoModelForImageTextToText, AutoProcessor

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
            repo_id="docling-project/CodeFormulaV2",
            revision="main",
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
        return self.enabled and (
            (isinstance(element, CodeItem) and self.options.do_code_enrichment)
            or (
                isinstance(element, TextItem)
                and element.label == DocItemLabel.FORMULA
                and self.options.do_formula_enrichment
            )
        )

    def _extract_code_language(self, input_string: str) -> Tuple[str, Optional[str]]:
        """Extracts a programming language from the beginning of a string.

        This function checks if the input string starts with a pattern of the form
        ``<_some_language_>``. If it does, it extracts the language string and returns
        a tuple of (remainder, language). Otherwise, it returns the original string
        and `None`.

        Args:
            input_string (str): The input string, which may start with ``<_language_>``.

        Returns:
            Tuple[str, Optional[str]]:
                A tuple where:
                - The first element is either:
                    - The remainder of the string (everything after ``<_language_>``),
                    if a match is found; or
                    - The original string, if no match is found.
                - The second element is the extracted language if a match is found;
                otherwise, `None`.
        """
        pattern = r"^<_([^_>]+)_>\s*(.*)"
        match = re.match(pattern, input_string, flags=re.DOTALL)
        if match:
            language = str(match.group(1))  # the captured programming language
            remainder = str(match.group(2))  # everything after the <_language_>
            return remainder, language
        else:
            return input_string, None

    def _get_code_language_enum(self, value: Optional[str]) -> CodeLanguageLabel:
        """
        Converts a string to a corresponding `CodeLanguageLabel` enum member.

        If the provided string does not match any value in `CodeLanguageLabel`,
        it defaults to `CodeLanguageLabel.UNKNOWN`.

        Args:
            value (Optional[str]): The string representation of the code language or None.

        Returns:
            CodeLanguageLabel: The corresponding enum member if the value is valid,
            otherwise `CodeLanguageLabel.UNKNOWN`.
        """
        if not isinstance(value, str):
            return CodeLanguageLabel.UNKNOWN

        try:
            return CodeLanguageLabel(value)
        except ValueError:
            return CodeLanguageLabel.UNKNOWN

    def _get_prompt(self, label: str) -> str:
        """
        Constructs the prompt for the model based on the input label.

        Parameters
        ----------
        label : str
            The type of input, either 'code' or 'formula'.

        Returns
        -------
        str
            The constructed prompt including necessary tokens and query.

        Raises
        ------
        NotImplementedError
            If the label is not 'code' or 'formula'.
        """
        if label == "code":
            query = "<code>"
        elif label == "formula":
            query = "<formula>"
        else:
            raise NotImplementedError("Label must be either code or formula")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": query}],
            },
        ]

        prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        return prompt

    def _post_process(self, texts: list[str]) -> list[str]:
        """
        Processes a list of text strings by truncating at '<end_of_utterance>' and
        removing a predefined set of unwanted substrings.

        Parameters
        ----------
        texts : list[str]
            A list of strings to be post-processed.

        Returns
        -------
        list[str]
            A list of cleaned strings with specified substrings removed and truncated at
                '<end_of_utterance>' if present.
        """
        to_remove = ["</code>", "</formula>", "<loc_0><loc_0><loc_500><loc_500>"]

        def clean_text(text: str) -> str:
            idx = text.find("<end_of_utterance>")
            if idx != -1:
                text = text[:idx]

            for token in to_remove:
                if token in text:
                    text = text.replace(token, "")
            return text.lstrip()

        return [clean_text(t) for t in texts]

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

        labels: List[str] = []
        images: List[Union[Image.Image, np.ndarray]] = []
        elements: List[TextItem] = []
        for el in element_batch:
            elements.append(el.item)  # type: ignore[arg-type]
            labels.append(el.item.label)  # type: ignore[attr-defined]
            images.append(el.image)

        prompts = [self._get_prompt(label) for label in labels]
        inputs = self._processor(
            text=prompts,
            images=images,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        gen_kwargs = dict(
            max_new_tokens=self._model_max_length - inputs.input_ids.shape[1],
            use_cache=True,
            do_sample=False,
        )

        generated_ids = self._model.generate(**inputs, **gen_kwargs)

        outputs = self._processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=False
        )
        outputs = self._post_process(outputs)

        for item, output in zip(elements, outputs):
            if isinstance(item, CodeItem):
                output, code_language = self._extract_code_language(output)
                item.code_language = self._get_code_language_enum(code_language)
            item.text = output

            yield item
