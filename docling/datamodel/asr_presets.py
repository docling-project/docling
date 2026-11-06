# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""ASR model specifications with engine-specific checkpoint exports."""

import importlib.metadata
import logging
import platform
import re
from typing import Final

from pydantic import BaseModel, Field

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options_asr_model import (
    AsrModelFamily,
    InferenceAsrFramework,
    InlineAsrMlxAudioOptions,
    InlineAsrMlxWhisperOptions,
    InlineAsrOptions,
    InlineAsrTransformersOptions,
)

_log = logging.getLogger(__name__)

# The backend that the `mlx` alias selects. This is the only place to change it.
_MLX_ALIAS_TARGET: Final = InferenceAsrFramework.MLX_WHISPER
# Keep this equal to the lower limit in the `feat-asr-mlx-audio` extra. `mlx-vlm` also
# installs `mlx-audio`, with a lower limit that this backend cannot use.
_MIN_MLX_AUDIO_VERSION: Final = "0.6.4"


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mlx_audio_is_usable() -> bool:
    """Tell if this machine can run mlx-audio, without an import of the library.

    Importing `mlx` initializes Metal, which can abort a restricted process. Only the
    model constructor can do that, so this check reads the package metadata.
    """
    if not is_apple_silicon():
        return False
    try:
        installed = importlib.metadata.version("mlx-audio")
    except importlib.metadata.PackageNotFoundError:
        return False
    # This runs on the import path of every install, which does not have `packaging`.
    # Only the release numbers matter here, so a pre-release counts as its release.
    release = re.match(r"\d+(?:\.\d+)*", installed)
    if release is None or tuple(
        int(number) for number in release.group().split(".")
    ) < tuple(int(number) for number in _MIN_MLX_AUDIO_VERSION.split(".")):
        _log.info(
            f"mlx-audio {installed} is older than {_MIN_MLX_AUDIO_VERSION}, "
            "so the ASR presets do not select it."
        )
        return False
    return True


def _resolve_framework(framework: InferenceAsrFramework) -> InferenceAsrFramework:
    """Resolve the `mlx` alias to a concrete MLX backend."""
    if framework == InferenceAsrFramework.MLX:
        return _MLX_ALIAS_TARGET
    return framework


class AsrEngineConfig(BaseModel):
    """Checkpoint settings for one ASR engine."""

    repo_id: str | None = None
    revision: str | None = None
    torch_dtype: str | None = None

    def merge_with(self, repo_id: str, revision: str) -> "AsrEngineConfig":
        return AsrEngineConfig(
            repo_id=self.repo_id or repo_id,
            revision=self.revision or revision,
            torch_dtype=self.torch_dtype,
        )


class AsrModelSpec(BaseModel):
    """Checkpoint locations supported by an ASR model family."""

    name: str
    default_repo_id: str
    revision: str = "main"
    model_family: AsrModelFamily = "whisper"
    supported_engines: set[InferenceAsrFramework]
    engine_overrides: dict[InferenceAsrFramework, AsrEngineConfig] = Field(
        default_factory=dict
    )

    def get_engine_config(self, engine: InferenceAsrFramework) -> AsrEngineConfig:
        resolved = _resolve_framework(engine)
        if resolved not in self.supported_engines:
            supported = ", ".join(sorted(e.value for e in self.supported_engines))
            raise ValueError(
                f"{self.name} does not support the {engine.value} engine. "
                f"Supported engines: {supported}."
            )
        override = self.engine_overrides.get(resolved, AsrEngineConfig())
        return override.merge_with(self.default_repo_id, self.revision)


class AsrModelPreset(BaseModel):
    """A named ASR model and its default engine."""

    preset_id: str
    model_spec: AsrModelSpec
    default_engine_type: InferenceAsrFramework

    def to_auto_options(
        self, device: AcceleratorDevice | str = AcceleratorDevice.AUTO
    ) -> InlineAsrOptions:
        """Return the options for the best engine on this machine.

        Select mlx-audio on Apple Silicon when it is usable, the model has an export
        for it, and the device allows Metal. Otherwise use the default engine.

        This runs when a process imports the SDK constants, so it cannot know the final
        device or test Metal. The selected options keep the default engine, and the
        model factory uses it when mlx-audio cannot run.
        """
        default = self.to_options()
        if (
            InferenceAsrFramework.MLX_AUDIO in self.model_spec.supported_engines
            and device in {AcceleratorDevice.AUTO, AcceleratorDevice.MPS}
            and _mlx_audio_is_usable()
        ):
            return self._to_mlx_audio_options(
                default if isinstance(default, InlineAsrTransformersOptions) else None
            )
        return default

    def _to_mlx_audio_options(
        self, transformers_fallback: InlineAsrTransformersOptions | None = None
    ) -> InlineAsrMlxAudioOptions:
        config = self.model_spec.get_engine_config(InferenceAsrFramework.MLX_AUDIO)
        if config.repo_id is None:
            raise ValueError(f"No repository ID for {self.preset_id}")
        return InlineAsrMlxAudioOptions(
            repo_id=config.repo_id,
            revision=config.revision or "main",
            model_family=self.model_spec.model_family,
            transformers_fallback=transformers_fallback,
        )

    def to_options(
        self, engine: InferenceAsrFramework | None = None
    ) -> InlineAsrOptions:
        selected = engine or self.default_engine_type
        config = self.model_spec.get_engine_config(selected)
        if config.repo_id is None:
            raise ValueError(f"No repository ID for {self.preset_id}")
        resolved = _resolve_framework(selected)
        if resolved == InferenceAsrFramework.TRANSFORMERS:
            return InlineAsrTransformersOptions(
                repo_id=config.repo_id,
                revision=config.revision or "main",
                torch_dtype=config.torch_dtype,
                model_family=self.model_spec.model_family,
            )
        if resolved == InferenceAsrFramework.MLX_WHISPER:
            return InlineAsrMlxWhisperOptions(
                repo_id=config.repo_id,
                # Keep the alias, because it is the value that saved options contain.
                inference_framework=(
                    InferenceAsrFramework.MLX
                    if selected == InferenceAsrFramework.MLX
                    else InferenceAsrFramework.MLX_WHISPER
                ),
            )
        if resolved == InferenceAsrFramework.MLX_AUDIO:
            return self._to_mlx_audio_options()
        raise ValueError(f"{selected.value} is not configured by this ASR preset")
