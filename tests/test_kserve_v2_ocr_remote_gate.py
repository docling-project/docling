# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the `enable_remote_services` check in the KServe v2 OCR model."""

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import KserveV2OcrOptions
from docling.exceptions import OperationNotAllowed
from docling.models.stages.ocr.kserve_v2_ocr_model import KserveV2OcrModel


def _options() -> KserveV2OcrOptions:
    # Valid URL that is never contacted.
    return KserveV2OcrOptions(url="http://kserve.invalid:8000")


def test_enabled_without_remote_services_raises(monkeypatch) -> None:
    def _boom(self) -> None:
        raise AssertionError("connection attempted before the remote-services gate")

    monkeypatch.setattr(KserveV2OcrModel, "_initialize_client", _boom)

    with pytest.raises(OperationNotAllowed):
        KserveV2OcrModel(
            enabled=True,
            artifacts_path=None,
            options=_options(),
            accelerator_options=AcceleratorOptions(),
            enable_remote_services=False,
        )


def test_default_is_deny() -> None:
    with pytest.raises(OperationNotAllowed):
        KserveV2OcrModel(
            enabled=True,
            artifacts_path=None,
            options=_options(),
            accelerator_options=AcceleratorOptions(),
        )


def test_disabled_does_not_raise(monkeypatch) -> None:
    monkeypatch.setattr(KserveV2OcrModel, "_initialize_client", lambda self: None)
    model = KserveV2OcrModel(
        enabled=False,
        artifacts_path=None,
        options=_options(),
        accelerator_options=AcceleratorOptions(),
        enable_remote_services=False,
    )
    assert model._kserve_client is None
