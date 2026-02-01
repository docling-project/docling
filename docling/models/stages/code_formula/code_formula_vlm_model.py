"""Code and formula extraction stage using the new VLM runtime system.

This module provides a runtime-agnostic code and formula extraction stage that can use
any VLM runtime (Transformers, MLX, API, etc.) through the unified runtime interface.
"""

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Tuple, Union

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

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.pipeline_options import CodeFormulaVlmOptions
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.models.runtimes.base import BaseVlmRuntime, VlmRuntimeInput
from docling.models.runtimes.factory import create_vlm_runtime

_log = logging.getLogger(__name__)


class CodeFormulaVlmModel(BaseItemAndImageEnrichmentModel):
    """Code and formula extraction stage using the new runtime system.

    This stage uses the unified VLM runtime interface to extract code and formulas
    from document elements. It supports all runtime types (Transformers, MLX,
    API, etc.) through the runtime factory.

    The stage:
    1. Filters code and formula elements
    2. Uses the runtime to extract text content
    3. Post-processes outputs (language detection for code, cleanup)
    4. Updates element text and metadata

    Example:
        ```python
        from docling.datamodel.pipeline_options import CodeFormulaVlmOptions

        # Use preset with default runtime
        options = CodeFormulaVlmOptions.from_preset("codeformulav2")

        # Create stage
        stage = CodeFormulaVlmModel(
            enabled=True,
            artifacts_path=None,
            options=options,
            accelerator_options=AcceleratorOptions(),
        )
        ```
    """

    elements_batch_size = 5
    images_scale = 1.67  # = 120 dpi, aligned with training data resolution
    expansion_factor = 0.18

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: CodeFormulaVlmOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """Initialize the code/formula extraction stage.

        Args:
            enabled: Whether this stage is enabled
            artifacts_path: Path to model artifacts (optional)
            options: Configuration options including model spec and runtime options
            accelerator_options: Hardware acceleration options
        """
        self.enabled = enabled
        self.options = options
        self.runtime: Optional[BaseVlmRuntime] = None

        if self.enabled:
            # Check if using new runtime system
            if (
                self.options.model_spec is not None
                and self.options.runtime_options is not None
            ):
                # New runtime system path
                runtime_type = self.options.runtime_options.runtime_type

                # Get model configuration for this runtime
                self.repo_id = self.options.model_spec.get_repo_id(runtime_type)
                self.revision = self.options.model_spec.get_revision(runtime_type)

                _log.info(
                    f"Initializing CodeFormulaVlmModel with runtime system: "
                    f"model={self.repo_id}, "
                    f"runtime={runtime_type.value}"
                )

                # Create runtime using factory
                self.runtime = create_vlm_runtime(
                    self.options.runtime_options, model_spec=self.options.model_spec
                )

                _log.info("CodeFormulaVlmModel initialized successfully")

            else:
                # Legacy path - fall back to old implementation
                raise ValueError(
                    "CodeFormulaVlmModel requires model_spec and runtime_options. "
                    "Use CodeFormulaVlmOptions.from_preset() to create options."
                )

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        """Determine if an element can be processed by this stage.

        Args:
            doc: The document being processed
            element: The element to check

        Returns:
            True if the element is a code block or formula that should be processed
        """
        return self.enabled and (
            (isinstance(element, CodeItem) and self.options.extract_code)
            or (
                isinstance(element, TextItem)
                and element.label == DocItemLabel.FORMULA
                and self.options.extract_formulas
            )
        )

    def _get_prompt(self, label: str) -> str:
        """Construct the prompt for the model based on the element type.

        Args:
            label: The type of input, either 'code' or 'formula'

        Returns:
            The prompt string

        Raises:
            NotImplementedError: If the label is not 'code' or 'formula'
        """
        if label == "code":
            return "<code>"
        elif label == "formula":
            return "<formula>"
        else:
            raise NotImplementedError("Label must be either code or formula")

    def _extract_code_language(self, input_string: str) -> Tuple[str, Optional[str]]:
        """Extract programming language from the beginning of a string.

        Checks if the input string starts with a pattern of the form
        ``<_some_language_>``. If it does, extracts the language string.

        Args:
            input_string: The input string, which may start with ``<_language_>``

        Returns:
            Tuple of (remainder, language) where:
            - remainder is the string after the language tag (or original if no match)
            - language is the extracted language if found, otherwise None
        """
        pattern = r"^<_([^_>]+)_>\s*(.*)"
        match = re.match(pattern, input_string, flags=re.DOTALL)
        if match:
            language = str(match.group(1))
            remainder = str(match.group(2))
            return remainder, language
        else:
            return input_string, None

    def _get_code_language_enum(self, value: Optional[str]) -> CodeLanguageLabel:
        """Convert a string to a CodeLanguageLabel enum member.

        Args:
            value: The string representation of the code language or None

        Returns:
            The corresponding enum member if valid, otherwise CodeLanguageLabel.UNKNOWN
        """
        if not isinstance(value, str):
            return CodeLanguageLabel.UNKNOWN

        try:
            return CodeLanguageLabel(value)
        except ValueError:
            return CodeLanguageLabel.UNKNOWN

    def _post_process(self, texts: list[str]) -> list[str]:
        """Post-process model outputs by removing unwanted tokens.

        Args:
            texts: List of strings to be post-processed

        Returns:
            List of cleaned strings with specified substrings removed
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
        """Process a batch of code/formula elements.

        Args:
            doc: The document being processed
            element_batch: Batch of elements to process

        Yields:
            Enriched elements with extracted text
        """
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        if self.runtime is None:
            raise RuntimeError("Runtime not initialized")

        labels: List[str] = []
        images: List[Union[Image.Image, np.ndarray]] = []
        elements: List[Union[CodeItem, TextItem]] = []

        for el in element_batch:
            assert isinstance(el.item, CodeItem | TextItem)
            elements.append(el.item)
            labels.append(el.item.label)
            images.append(el.image)

        # Process batch through runtime
        try:
            # Prepare batch of runtime inputs
            runtime_inputs = [
                VlmRuntimeInput(
                    image=image
                    if isinstance(image, Image.Image)
                    else Image.fromarray(image),
                    prompt=self._get_prompt(label),
                    temperature=0.0,
                    max_new_tokens=2048,
                )
                for image, label in zip(images, labels)
            ]

            # Run batch inference
            batch_outputs = self.runtime.predict_batch(runtime_inputs)
            outputs = [output.text for output in batch_outputs]

        except Exception as e:
            _log.error(f"Error processing code/formula batch: {e}")
            outputs = [""] * len(images)

        # Post-process outputs
        outputs = self._post_process(outputs)

        # Update elements with extracted text
        for item, output_text in zip(elements, outputs):
            if isinstance(item, CodeItem):
                output_text, code_language = self._extract_code_language(output_text)
                item.code_language = self._get_code_language_enum(code_language)
            item.text = output_text

            yield item

    def __del__(self):
        """Cleanup runtime resources."""
        if self.runtime is not None:
            try:
                self.runtime.cleanup()
            except Exception as e:
                _log.warning(f"Error cleaning up runtime: {e}")
