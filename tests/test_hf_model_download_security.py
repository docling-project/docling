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
    """Every trust_remote_code spec shipped with docling must be commit-pinned."""

    def test_stage_presets(self):
        from docling.datamodel import stage_model_specs
        from docling.datamodel.stage_model_specs import StageModelPreset

        presets = _module_level(stage_model_specs, StageModelPreset)
        assert presets
        for name, preset in presets:
            spec = preset.model_spec
            if not getattr(spec, "trust_remote_code", False):
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
        from docling.pipeline.asr_transcriber import (
            _DISTIL_WHISPER_OPENAI_CHECKPOINTS,
        )

        for name, (_, _, revision) in _DISTIL_WHISPER_OPENAI_CHECKPOINTS.items():
            assert is_pinned_revision(revision), name
