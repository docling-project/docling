# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import warnings
from typing import Any, ClassVar, Dict, Literal, Optional

from pydantic import Field, model_validator
from typing_extensions import Self

from docling.datamodel.stage_model_specs import StagePresetMixin, VlmModelSpec
from docling.models.inference_engines.vlm.base import VlmEngineOptionsMixin


class ChartExtractionVlmEngineOptions(StagePresetMixin, VlmEngineOptionsMixin):
    """Configuration for the chart extraction enrichment stage.

    Uses the unified VLM engine system (Transformers / API / MLX / vLLM) and
    the same preset mechanism as picture description and code/formula stages.

    The three output modes are independent; enable the ones you need:

    * ``chart2csv``     — extract numeric data as a CSV table (default: True)
    * ``chart2summary`` — generate a natural-language description (default: False)
    * ``chart2code``    — generate Python code that recreates the chart (default: False)

    Examples::

        # Default preset (granite_vision_v4, Transformers engine)
        options = ChartExtractionVlmEngineOptions.from_preset("granite_vision_v4")

        # Legacy V1 model
        options = ChartExtractionVlmEngineOptions.from_preset("granite_vision")

        # Override engine at preset time
        from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
        options = ChartExtractionVlmEngineOptions.from_preset(
            "granite_vision_v4",
            engine_options=ApiVlmEngineOptions(
                engine_type=VlmEngineType.API_OPENAI,
                url="http://localhost:8000/v1/chat/completions",
            ),
        )
    """

    kind: ClassVar[Literal["chart_extraction_vlm_engine"]] = (
        "chart_extraction_vlm_engine"
    )

    model_spec: VlmModelSpec = Field(
        description="Model specification with engine-specific overrides"
    )

    chart2csv: bool = Field(
        default=True,
        description=(
            "Extract numeric data from the chart as a CSV table with headers and values."
        ),
    )
    chart2summary: bool = Field(
        default=False,
        description=(
            "Generate a natural-language summary describing the chart."
        ),
    )
    chart2code: bool = Field(
        default=False,
        description=(
            "Generate Python code that recreates the chart."
        ),
    )

    @model_validator(mode="after")
    def _at_least_one_output(self) -> Self:
        if not (self.chart2csv or self.chart2summary or self.chart2code):
            raise ValueError(
                "At least one of chart2csv, chart2summary, or chart2code must be True."
            )
        return self

    def active_prompts(self) -> list[str]:
        """Return the ordered list of special-token prompts to send for each chart image.

        The V4 model dispatches on these tokens; the V1 model uses a fixed prompt
        from the model spec and ignores this method.
        """
        prompts: list[str] = []
        if self.chart2csv:
            prompts.append("<chart2csv>")
        if self.chart2summary:
            prompts.append("<chart2summary>")
        if self.chart2code:
            prompts.append("<chart2code>")
        return prompts


# ---------------------------------------------------------------------------
# Deprecated shims — kept for backwards compatibility only
# ---------------------------------------------------------------------------


class ChartExtractionModelKind:
    """Deprecated — use ``ChartExtractionVlmEngineOptions.from_preset()`` instead.

    .. deprecated::
        Use :meth:`ChartExtractionVlmEngineOptions.from_preset` with
        ``'granite_vision'`` or ``'granite_vision_v4'`` instead.
    """

    GRANITE_VISION = "granite-vision"
    GRANITE_VISION_V4 = "granite-vision-v4"

    # Map old enum values to new preset IDs
    _PRESET_MAP: ClassVar[Dict[str, str]] = {
        "granite-vision": "granite_vision",
        "granite-vision-v4": "granite_vision_v4",
    }


class ChartExtractionModelOptions(ChartExtractionVlmEngineOptions):
    """Deprecated — use ``ChartExtractionVlmEngineOptions`` instead.

    For backwards compatibility, instantiating this class emits a
    ``DeprecationWarning`` and returns a fully functional
    ``ChartExtractionVlmEngineOptions`` configured from the ``granite_vision_v4``
    preset (or ``granite_vision`` when ``model=ChartExtractionModelKind.GRANITE_VISION``
    was passed).
    """

    kind: ClassVar[Literal["chart_extraction"]] = "chart_extraction"  # type: ignore[assignment]

    # The old 'model' field is accepted but ignored after mapping to a preset.
    model: Optional[str] = Field(
        default=None,
        description=(
            "Deprecated. Use ChartExtractionVlmEngineOptions.from_preset() instead."
        ),
        exclude=True,
    )

    def __init__(self, **data: Any) -> None:
        warnings.warn(
            "ChartExtractionModelOptions is deprecated. "
            "Use ChartExtractionVlmEngineOptions.from_preset('granite_vision_v4') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Resolve preset from the legacy 'model' field if provided.
        model_val = data.pop("model", None)
        preset_id = "granite_vision_v4"
        if model_val is not None:
            preset_id = ChartExtractionModelKind._PRESET_MAP.get(
                str(model_val), "granite_vision_v4"
            )

        # Bootstrap from the preset so model_spec and engine_options are populated,
        # then allow the caller's remaining kwargs (chart2csv, etc.) to override.
        preset_instance = ChartExtractionVlmEngineOptions.from_preset(preset_id)
        merged = {**preset_instance.model_dump(), **data}
        super().__init__(**merged)
