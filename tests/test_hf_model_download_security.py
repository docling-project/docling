# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Security-guard tests for the model-download revision-pinning helpers.

These cover the moving-ref detector and the trust_remote_code supply-chain guard
that warns (or refuses) when custom code from the Hub would be loaded from an
unpinned revision. See docling.models.utils.hf_model_download.
"""

import logging

import pytest

from docling.datamodel.settings import settings
from docling.models.utils.hf_model_download import (
    is_pinned_revision,
    warn_on_unpinned_trust_remote_code,
)

_PINNED_SHA = "0123456789abcdef0123456789abcdef01234567"  # 40 hex chars


class TestIsPinnedRevision:
    def test_full_commit_sha_is_pinned(self):
        assert is_pinned_revision(_PINNED_SHA) is True

    def test_uppercase_sha_is_pinned(self):
        assert is_pinned_revision(_PINNED_SHA.upper()) is True

    def test_sha_with_surrounding_whitespace_is_pinned(self):
        assert is_pinned_revision(f"  {_PINNED_SHA}\n") is True

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
        monkeypatch.setattr(
            settings.security, "refuse_unpinned_remote_code", True, raising=False
        )
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError, match="SECURITY"):
                warn_on_unpinned_trust_remote_code(
                    "some/repo", revision="main", trust_remote_code=True
                )
        # Refusal raises rather than logging a warning.
        assert caplog.records == []

    def test_refuse_flag_allows_pinned_revision(self, monkeypatch):
        monkeypatch.setattr(
            settings.security, "refuse_unpinned_remote_code", True, raising=False
        )
        # A pinned revision must not raise even when refusal is enabled.
        warn_on_unpinned_trust_remote_code(
            "some/repo", revision=_PINNED_SHA, trust_remote_code=True
        )
