# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the revision-pinning helpers in ``hf_model_download``.

These cover commit-SHA detection and the check that warns (or refuses) when a
``trust_remote_code`` model is downloaded from an unpinned revision.
"""

import logging

import pytest

from docling.models.utils.hf_model_download import (
    _REFUSE_ENV_VAR,
    is_pinned_revision,
    warn_on_unpinned_trust_remote_code,
)

_PINNED_SHA = "0123456789abcdef0123456789abcdef01234567"  # 40 hex chars


class TestIsPinnedRevision:
    def test_full_commit_sha_is_pinned(self):
        assert is_pinned_revision(_PINNED_SHA) is True

    @pytest.mark.parametrize(
        "revision",
        [
            _PINNED_SHA.upper(),
            f"  {_PINNED_SHA}",
            f"{_PINNED_SHA}\n",
        ],
    )
    def test_sha_not_as_passed_to_hub_is_not_pinned(self, revision):
        assert is_pinned_revision(revision) is False

    @pytest.mark.parametrize(
        "revision",
        [
            None,
            "main",
            "master",
            "v1.0",
            "v1.0.0",
            "my-branch",
            "0123456789abcdef",  # short SHA, not 40 chars
            "0123456789abcdef0123456789abcdef0123456g",  # 40 chars, non-hex 'g'
        ],
    )
    def test_moving_refs_are_not_pinned(self, revision):
        assert is_pinned_revision(revision) is False


class TestWarnOnUnpinnedTrustRemoteCode:
    def test_no_warning_when_trust_remote_code_false(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_on_unpinned_trust_remote_code(
                "some/repo", revision=None, trust_remote_code=False
            )
        assert caplog.records == []

    def test_no_warning_when_pinned_even_with_trust_remote_code(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_on_unpinned_trust_remote_code(
                "some/repo", revision=_PINNED_SHA, trust_remote_code=True
            )
        assert caplog.records == []

    @pytest.mark.parametrize("revision", [None, "main", "v1.0", "a-branch"])
    def test_warns_on_moving_ref_with_trust_remote_code(self, caplog, revision):
        with caplog.at_level(logging.WARNING):
            warn_on_unpinned_trust_remote_code(
                "some/repo", revision=revision, trust_remote_code=True
            )
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.WARNING
        assert "SECURITY" in record.getMessage()
        assert "some/repo" in record.getMessage()

    def test_refuse_flag_raises_instead_of_warning(self, caplog, monkeypatch):
        monkeypatch.setenv(_REFUSE_ENV_VAR, "1")
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError, match="SECURITY"):
                warn_on_unpinned_trust_remote_code(
                    "some/repo", revision="main", trust_remote_code=True
                )
        # Refusal raises rather than logging a warning.
        assert caplog.records == []

    def test_refuse_flag_allows_pinned_revision(self, monkeypatch):
        monkeypatch.setenv(_REFUSE_ENV_VAR, "1")
        # A pinned revision must not raise even when refusal is enabled.
        warn_on_unpinned_trust_remote_code(
            "some/repo", revision=_PINNED_SHA, trust_remote_code=True
        )

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "  "])
    def test_non_truthy_env_values_only_warn(self, caplog, monkeypatch, value):
        monkeypatch.setenv(_REFUSE_ENV_VAR, value)
        with caplog.at_level(logging.WARNING):
            warn_on_unpinned_trust_remote_code(
                "some/repo", revision="main", trust_remote_code=True
            )
        assert len(caplog.records) == 1


def _module_level(module, cls):
    return [(name, obj) for name, obj in vars(module).items() if isinstance(obj, cls)]


class TestShippedSpecsArePinned:
    """Every spec shipped with docling that can run remote code must be pinned."""

    def test_stage_presets(self):
        from docling.datamodel import stage_model_specs
        from docling.datamodel.stage_model_specs import StageModelPreset

        presets = _module_level(stage_model_specs, StageModelPreset)
        assert presets
        for name, preset in presets:
            spec = preset.model_spec
            # Presets with a user opt-in gate also run remote code once enabled.
            requires_opt_in = any(
                override.extra_config.get("requires_trust_remote_code")
                for override in spec.engine_overrides.values()
            )
            if not (getattr(spec, "trust_remote_code", False) or requires_opt_in):
                continue
            assert is_pinned_revision(spec.revision), name
            for engine, override in spec.engine_overrides.items():
                assert is_pinned_revision(spec.get_revision(engine)), (
                    name,
                    engine,
                    override.repo_id,
                )

    def test_inline_vlm_options(self):
        from docling.datamodel import vlm_model_specs
        from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions

        specs = _module_level(vlm_model_specs, InlineVlmOptions)
        assert specs
        for name, spec in specs:
            if spec.trust_remote_code:
                assert is_pinned_revision(spec.revision), name

    def test_distil_whisper_checkpoints(self):
        from docling.datamodel.asr_model_specs import (
            _DISTIL_WHISPER_OPENAI_CHECKPOINTS,
        )

        for name, (_, _, revision) in _DISTIL_WHISPER_OPENAI_CHECKPOINTS.items():
            assert is_pinned_revision(revision), name


@pytest.mark.parametrize(
    "model_name",
    ["distil-small.en", "distil-medium.en", "distil-large-v3", "distil-large-v3.5"],
)
@pytest.mark.parametrize("offline", [False, True])
def test_distil_whisper_prefetch_matches_runtime_revision(
    tmp_path, model_name, offline
):
    from unittest.mock import Mock, patch

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.asr_model_specs import _DISTIL_WHISPER_OPENAI_CHECKPOINTS
    from docling.datamodel.pipeline_options_asr_model import (
        InlineAsrNativeWhisperOptions,
    )
    from docling.pipeline.asr_transcriber import _NativeWhisperModel
    from docling.utils.model_downloader import _download_distil_whisper_models

    repo_id, filename, revision = _DISTIL_WHISPER_OPENAI_CHECKPOINTS[model_name]
    checkpoint = str(tmp_path / filename)
    with (
        patch("huggingface_hub.hf_hub_download", return_value=checkpoint) as download,
        patch.dict("sys.modules", {"whisper": Mock()}),
    ):
        _download_distil_whisper_models(
            [model_name], tmp_path, force=False, progress=False
        )
        download.assert_called_once_with(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=tmp_path,
            force_download=False,
        )
        download.reset_mock()

        _NativeWhisperModel(
            enabled=True,
            artifacts_path=tmp_path if offline else None,
            accelerator_options=AcceleratorOptions(device="cpu"),
            asr_options=InlineAsrNativeWhisperOptions(repo_id=model_name),
        )
        expected = dict(repo_id=repo_id, filename=filename, revision=revision)
        if offline:
            expected.update(cache_dir=str(tmp_path), local_files_only=True)
        download.assert_called_once_with(**expected)
