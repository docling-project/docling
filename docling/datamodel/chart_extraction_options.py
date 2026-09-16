# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import warnings
from typing import Any, ClassVar, Dict, Iterator, Literal, Optional

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

    .. note::
        The ``granite_vision`` V1 preset (ibm-granite/granite-vision-3.3-2b-chart2csv-preview)
        was removed in this release. Use ``granite_vision_v4`` (the default) instead.
        The last release supporting V1 was 2.x (see the changelog for migration guidance).

    Examples::

        # Default preset (granite_vision_v4, Transformers engine)
        options = ChartExtractionVlmEngineOptions.from_preset("granite_vision_v4")

        # Override engine at preset time
        from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
        options = ChartExtractionVlmEngineOptions.from_preset(
            "granite_vision_v4",
            engine_options=ApiVlmEngineOptions(
                engine_type=VlmEngineType.API_OPENAI,
                url="http://localhost:8000/v1/chat/completions",
            ),
        )

        # Serving a GGUF quantization that lacks the fine-tuned special tokens?
        # Enable natural-language prompt substitution:
        options = ChartExtractionVlmEngineOptions.from_preset(
            "granite_vision_v4",
            engine_options=ApiVlmEngineOptions(
                engine_type=VlmEngineType.API_LMSTUDIO,
                url="http://localhost:1234/v1/chat/completions",
            ),
            use_natural_language_prompts=True,
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
        description=("Generate a natural-language summary describing the chart."),
    )
    chart2code: bool = Field(
        default=False,
        description=("Generate Python code that recreates the chart."),
    )
    use_natural_language_prompts: bool = Field(
        default=False,
        description=(
            "Replace special-token prompts (<chart2csv>, <chart2summary>, <chart2code>) "
            "with natural-language equivalents. Enable this when the deployed model does "
            "not have the fine-tuned special tokens (e.g. a GGUF quantization served via "
            "an API endpoint that lacks HF fine-tuning)."
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
        """Return the ordered list of special-token prompts to send for each chart image."""
        prompts: list[str] = []
        if self.chart2csv:
            prompts.append("<chart2csv>")
        if self.chart2summary:
            prompts.append("<chart2summary>")
        if self.chart2code:
            prompts.append("<chart2code>")
        return prompts


# ---------------------------------------------------------------------------
# Register chart extraction presets alongside the options class so that
# from_preset() works even when pipeline_options has not been imported yet.
# ---------------------------------------------------------------------------

from docling.datamodel import stage_model_specs as _stage_model_specs  # noqa: E402

ChartExtractionVlmEngineOptions.register_preset(
    _stage_model_specs.CHART_EXTRACTION_GRANITE_VISION_V4
)


# ---------------------------------------------------------------------------
# Deprecated shims — kept for backwards compatibility only
# ---------------------------------------------------------------------------


class _ChartExtractionModelKindMeta(type):
    """Metaclass that makes ChartExtractionModelKind iterable and subscriptable
    so that legacy code using ``list(ChartExtractionModelKind)`` or
    ``ChartExtractionModelKind["GRANITE_VISION"]`` does not raise TypeError.
    """

    def __iter__(cls) -> Iterator[Any]:
        return iter(cls._members.values())

    def __getitem__(cls, item: str) -> Any:
        return cls._members[item]

    def __contains__(cls, item: object) -> bool:
        return item in cls._members.values()


class ChartExtractionModelKind(metaclass=_ChartExtractionModelKindMeta):
    """Deprecated — use ``ChartExtractionVlmEngineOptions.from_preset()`` instead.

    .. deprecated::
        Use :meth:`ChartExtractionVlmEngineOptions.from_preset` with
        ``'granite_vision_v4'`` instead.

    .. note::
        ``GRANITE_VISION`` (V1) support has been removed. References to
        ``ChartExtractionModelKind.GRANITE_VISION`` will resolve to
        ``'granite-vision-v4'`` with a deprecation warning.
    """

    GRANITE_VISION = "granite-vision"
    GRANITE_VISION_V4 = "granite-vision-v4"

    # Instances exposed as attributes so that `.value` works like an enum member
    class _Member:
        def __init__(self, name: str, value: str) -> None:
            self.name = name
            self.value = value

        def __repr__(self) -> str:
            return f"<ChartExtractionModelKind.{self.name}: {self.value!r}>"

        def __str__(self) -> str:
            return self.value

        def __eq__(self, other: object) -> bool:
            if isinstance(other, str):
                return self.value == other
            if isinstance(other, ChartExtractionModelKind._Member):
                return self.value == other.value
            return NotImplemented

        def __hash__(self) -> int:
            return hash(self.value)

    _members: ClassVar[Dict[str, "_ChartExtractionModelKindMeta"]] = {}

    # Map old enum values to new preset IDs (V1 → V4 with deprecation)
    _PRESET_MAP: ClassVar[Dict[str, str]] = {
        "granite-vision": "granite_vision_v4",
        "granite-vision-v4": "granite_vision_v4",
    }


# Populate _members after the class body is fully defined
for _name, _val in [
    ("GRANITE_VISION", ChartExtractionModelKind.GRANITE_VISION),
    ("GRANITE_VISION_V4", ChartExtractionModelKind.GRANITE_VISION_V4),
]:
    _member = ChartExtractionModelKind._Member(_name, _val)
    ChartExtractionModelKind._members[_name] = _member  # type: ignore[assignment]
    setattr(ChartExtractionModelKind, _name, _member)


class ChartExtractionModelOptions(ChartExtractionVlmEngineOptions):
    """Deprecated — use ``ChartExtractionVlmEngineOptions`` instead.

    For backwards compatibility, instantiating this class emits a
    ``DeprecationWarning`` and returns a fully functional
    ``ChartExtractionVlmEngineOptions`` configured from the ``granite_vision_v4``
    preset. Passing ``model=ChartExtractionModelKind.GRANITE_VISION`` (V1) is
    accepted but silently upgraded to V4 with an additional warning.
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
        if model_val is not None:
            model_str = str(model_val)
            if model_str not in ChartExtractionModelKind._PRESET_MAP:
                raise ValueError(
                    f"Unknown model {model_str!r}. "
                    f"Valid values: {list(ChartExtractionModelKind._PRESET_MAP)}"
                )
            if model_str == ChartExtractionModelKind.GRANITE_VISION:
                warnings.warn(
                    "ChartExtractionModelKind.GRANITE_VISION (V1) is no longer supported "
                    "and has been upgraded to granite_vision_v4.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # Bootstrap from the preset so model_spec and engine_options are populated,
        # then allow the caller's remaining kwargs (chart2csv, etc.) to override.
        preset_instance = ChartExtractionVlmEngineOptions.from_preset(
            "granite_vision_v4"
        )
        merged = {**preset_instance.model_dump(), **data}
        super().__init__(**merged)
