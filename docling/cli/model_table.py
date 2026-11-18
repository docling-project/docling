# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The data behind `docling models`: which backend can run which model."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec
from typing import Final

from docling.datamodel.asr_model_specs import AsrModelType
from docling.datamodel.asr_presets import is_apple_silicon
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.pipeline_options_asr_model import (
    InlineAsrMlxAudioOptions,
    InlineAsrMlxWhisperOptions,
    InlineAsrNativeWhisperOptions,
    InlineAsrOptions,
    InlineAsrTransformersOptions,
    InlineAsrWhisperS2TOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.datamodel.stage_model_specs import StageModelPreset
from docling.models.inference_engines.vlm.base import VlmEngineType


class BackendState(str, Enum):
    SELECTED = "selected"
    SUPPORTED = "supported"
    ALLOWED = "allowed"


@dataclass(frozen=True)
class ModelRow:
    name: str
    family: str
    backends: dict[str, BackendState]
    # What the model gives on the selected backend, by column key.
    features: dict[str, bool]


@dataclass(frozen=True)
class ModelTable:
    kind: str
    title: str
    # Column key, then the label in the table.
    backends: dict[str, str]
    # Column key, then the label in the table.
    features: dict[str, str]
    rows: list[ModelRow]
    default_model: str


# Column key, then the label in the table. Two-line labels keep the table in 80 columns.
ASR_BACKENDS: Final[dict[str, str]] = {
    "native": "Native",
    "mlx_whisper": "MLX\nWhisper",
    "mlx_audio": "mlx-\naudio",
    "transformers": "Trans-\nformers",
    "s2t": "S2T",
}
VLM_BACKENDS: Final[dict[str, str]] = {
    "mlx": "MLX",
    "transformers": "Trans-\nformers",
    "vllm": "vLLM",
    "api": "API",
}

ASR_FEATURES: Final[dict[str, str]] = {"timestamps": "Times", "speaker_labels": "Spk"}
VLM_FEATURES: Final[dict[str, str]] = {"bbox_grounding": "Bbox"}

# The formats whose parser keeps the boxes of the model in `DoclingDocument`. The others
# go through `_convert_text_page`, which gives each item an empty box. A test fails when
# a new format is in neither set.
BBOX_RESPONSE_FORMATS: Final = frozenset(
    {
        ResponseFormat.DOCTAGS,
        ResponseFormat.DOCLANG,
        ResponseFormat.DEEPSEEKOCR_MARKDOWN,
        ResponseFormat.UNLIMITED_OCR_MARKDOWN,
        ResponseFormat.UNLIMITED_OCR2,
        ResponseFormat.LIGHTONOCR_MARKDOWN,
        ResponseFormat.CHANDRA_HTML,
        ResponseFormat.SURYA_HTML,
        ResponseFormat.DOTS_JSON,
        ResponseFormat.NEMOTRON_PARSE_V2,
        ResponseFormat.NEMOTRON_PARSE_V3_5,
        ResponseFormat.MINERU2,
    }
)
NO_BBOX_RESPONSE_FORMATS: Final = frozenset(
    {
        ResponseFormat.MARKDOWN,
        ResponseFormat.HTML,
        ResponseFormat.OTSL,
        ResponseFormat.PLAINTEXT,
    }
)

_ASR_OPTIONS_COLUMNS: Final[dict[type[InlineAsrOptions], str]] = {
    InlineAsrNativeWhisperOptions: "native",
    InlineAsrMlxWhisperOptions: "mlx_whisper",
    InlineAsrMlxAudioOptions: "mlx_audio",
    InlineAsrTransformersOptions: "transformers",
    InlineAsrWhisperS2TOptions: "s2t",
}
# `_mlx` is the alias of `_mlx_whisper`.
_ASR_SUFFIXES: Final = (
    "_mlx_whisper",
    "_mlx_audio",
    "_transformers",
    "_native",
    "_s2t",
    "_mlx",
)
_ASR_FAMILIES: Final = (
    ("whisper", "Whisper"),
    ("granite", "Granite Speech"),
    ("parakeet", "Parakeet"),
    ("nemotron", "Nemotron"),
)
_VLM_FAMILIES: Final = (
    ("granite", "Granite"),
    ("smoldocling", "SmolDocling"),
    ("nemotron_parse", "Nemotron Parse"),
    ("chandra", "Datalab"),
    ("surya", "Datalab"),
    ("deepseek", "DeepSeek"),
    ("dolphin", "Dolphin"),
    ("dots", "dots"),
    ("falcon", "Falcon"),
    ("gemma", "Gemma"),
    ("glm", "GLM"),
    ("got_ocr", "GOT-OCR"),
    ("lightonocr", "LightOnOCR"),
    ("mineru", "MinerU"),
    ("nanonets", "Nanonets"),
    ("phi", "Phi"),
    ("pixtral", "Pixtral"),
    ("qwen", "Qwen"),
    ("unlimited_ocr", "Unlimited-OCR"),
)
# A new preset without an entry above goes to this group. Add an entry to name its family.
OTHER_FAMILY: Final = "Other"


def _family(name: str, families: tuple[tuple[str, str], ...]) -> str:
    for prefix, family in families:
        if name.startswith(prefix):
            return family
    return OTHER_FAMILY


def _in_family_order(
    rows: list[ModelRow], families: tuple[tuple[str, str], ...]
) -> list[ModelRow]:
    order = [family for _, family in families] + [OTHER_FAMILY]
    return sorted(rows, key=lambda row: order.index(row.family))


def _asr_features(options: InlineAsrOptions) -> dict[str, bool]:
    """Tell what the decoders of this repository give for these options.

    A time range that only covers a decoding window does not count as a timestamp.
    """
    if not isinstance(
        options, (InlineAsrMlxAudioOptions, InlineAsrTransformersOptions)
    ):
        # The Whisper backends without a model family.
        return {"timestamps": True, "speaker_labels": False}
    family = options.model_family
    if family == "granite_plus":
        return {
            "timestamps": options.word_timestamps,
            "speaker_labels": options.speaker_attribution,
        }
    timestamps = family == "whisper" or (
        family in {"parakeet", "nemotron"}
        and isinstance(options, InlineAsrMlxAudioOptions)
    )
    return {"timestamps": timestamps, "speaker_labels": False}


def asr_rows(resolve: Callable[[AsrModelType], InlineAsrOptions]) -> list[ModelRow]:
    """Build the rows from the presets, so that a new preset cannot be missing.

    A name without a backend suffix selects its backend for this machine, so its
    resolved backend is the selected one. A name with a suffix only shows support. The
    features come from the selected backend. A row without one needs a suffix, so it
    shows what any of its backends gives.
    """
    backends_of: dict[str, dict[str, BackendState]] = {}
    features_of: dict[str, dict[str, bool]] = {}
    for model in AsrModelType:
        base = model.value
        state = BackendState.SELECTED
        for suffix in _ASR_SUFFIXES:
            if base.endswith(suffix):
                base = base.removesuffix(suffix)
                state = BackendState.SUPPORTED
                break
        options = resolve(model)
        column = _ASR_OPTIONS_COLUMNS[type(options)]
        backends = backends_of.setdefault(base, {})
        has_selected = BackendState.SELECTED in backends.values()
        if backends.get(column) != BackendState.SELECTED:
            backends[column] = state
        features = _asr_features(options)
        if state == BackendState.SELECTED:
            features_of[base] = features
        elif not has_selected:
            known = features_of.get(base, {})
            features_of[base] = {
                key: value or known.get(key, False) for key, value in features.items()
            }
    rows = [
        ModelRow(name, _family(name, _ASR_FAMILIES), backends, features_of[name])
        for name, backends in backends_of.items()
    ]
    return _in_family_order(rows, _ASR_FAMILIES)


def _vlm_column(engine: VlmEngineType) -> str | None:
    if VlmEngineType.is_api_variant(engine):
        return "api"
    return {
        VlmEngineType.MLX: "mlx",
        VlmEngineType.TRANSFORMERS: "transformers",
        VlmEngineType.VLLM: "vllm",
    }.get(engine)


def _mlx_vlm_can_run(preset: StageModelPreset) -> bool:
    """Apply the MLX rule of `AutoInlineVlmEngine`, without torch or an MLX import."""
    spec = preset.model_spec
    if (
        not is_apple_silicon()
        or not spec.has_explicit_engine_export(VlmEngineType.MLX)
        or find_spec("mlx_vlm") is None
    ):
        return False
    # This module needs packages that mlx-vlm brings, so it comes after that check.
    from docling.models.inference_engines.vlm._utils import engine_version_satisfied

    override = spec.engine_overrides.get(VlmEngineType.MLX)
    return engine_version_satisfied(
        VlmEngineType.MLX, override.min_engine_version if override else None
    )


def _selected_vlm_column(preset: StageModelPreset) -> str | None:
    if preset.default_engine_type != VlmEngineType.AUTO_INLINE:
        return _vlm_column(preset.default_engine_type)
    return "mlx" if _mlx_vlm_can_run(preset) else "transformers"


def _vlm_row(preset_id: str, preset: StageModelPreset) -> ModelRow:
    spec = preset.model_spec
    backends: dict[str, BackendState] = {}
    declared = (
        spec.supported_engines
        if spec.supported_engines is not None
        else spec.engine_overrides
    )
    for engine in declared:
        column = _vlm_column(engine)
        if column is not None:
            backends[column] = BackendState.SUPPORTED
    if spec.supported_engines is None:
        # No restriction is not a statement that the backend works.
        for column in VLM_BACKENDS:
            backends.setdefault(column, BackendState.ALLOWED)
    selected = _selected_vlm_column(preset)
    if selected is not None:
        backends[selected] = BackendState.SELECTED
    return ModelRow(
        preset_id,
        _family(preset_id, _VLM_FAMILIES),
        backends,
        {"bbox_grounding": spec.response_format in BBOX_RESPONSE_FORMATS},
    )


def vlm_rows() -> list[ModelRow]:
    rows = [
        _vlm_row(preset_id, VlmConvertOptions.get_preset(preset_id))
        for preset_id in VlmConvertOptions.list_preset_ids()
    ]
    return _in_family_order(rows, _VLM_FAMILIES)
