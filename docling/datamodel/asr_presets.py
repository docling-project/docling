# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""ASR model specifications with engine-specific checkpoint exports."""

from typing import Literal

from pydantic import BaseModel, Field

from docling.datamodel.pipeline_options_asr_model import (
    InferenceAsrFramework,
    InlineAsrMlxWhisperOptions,
    InlineAsrOptions,
    InlineAsrTransformersOptions,
)


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
    model_family: Literal["whisper", "granite", "parakeet", "nemotron"] = "whisper"
    supported_engines: set[InferenceAsrFramework]
    engine_overrides: dict[InferenceAsrFramework, AsrEngineConfig] = Field(
        default_factory=dict
    )

    def get_engine_config(self, engine: InferenceAsrFramework) -> AsrEngineConfig:
        if engine not in self.supported_engines:
            raise ValueError(f"{self.name} does not support the {engine.value} engine")
        override = self.engine_overrides.get(engine, AsrEngineConfig())
        return override.merge_with(self.default_repo_id, self.revision)


class AsrModelPreset(BaseModel):
    """A named ASR model and its default engine."""

    preset_id: str
    model_spec: AsrModelSpec
    default_engine_type: InferenceAsrFramework

    def to_options(
        self, engine: InferenceAsrFramework | None = None
    ) -> InlineAsrOptions:
        selected = engine or self.default_engine_type
        config = self.model_spec.get_engine_config(selected)
        if config.repo_id is None:
            raise ValueError(f"No repository ID for {self.preset_id}")
        if selected == InferenceAsrFramework.TRANSFORMERS:
            return InlineAsrTransformersOptions(
                repo_id=config.repo_id,
                revision=config.revision or "main",
                torch_dtype=config.torch_dtype,
                model_family=self.model_spec.model_family,
            )
        if selected == InferenceAsrFramework.MLX:
            return InlineAsrMlxWhisperOptions(repo_id=config.repo_id)
        raise ValueError(f"{selected.value} is not configured by this ASR preset")
