# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The KServe v2 OCR model must honor `enable_remote_services`.

This engine ships page crops to a remote inference server, so -- like every
other remote engine in Docling -- it must refuse to run unless remote services
have been explicitly enabled, and it must refuse *before* opening any
connection.
"""

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import KserveV2OcrOptions
from docling.exceptions import OperationNotAllowed
from docling.models.stages.ocr.kserve_v2_ocr_model import KserveV2OcrModel


def _options() -> KserveV2OcrOptions:
    # A syntactically valid but never-contacted endpoint.
    return KserveV2OcrOptions(url="http://kserve.invalid:8000")


def test_enabled_without_remote_services_raises(monkeypatch) -> None:
    # If the guard failed to fire first, this would run and blow up loudly
    # instead of the expected OperationNotAllowed -- proving the guard runs
    # before any client/connection is created.
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
    # Omitting the flag entirely must also deny (deny-by-default).
    with pytest.raises(OperationNotAllowed):
        KserveV2OcrModel(
            enabled=True,
            artifacts_path=None,
            options=_options(),
            accelerator_options=AcceleratorOptions(),
        )


def test_disabled_does_not_raise(monkeypatch) -> None:
    # A disabled engine never connects, so the gate is irrelevant and
    # construction must succeed (mirrors the picture-description model).
    monkeypatch.setattr(KserveV2OcrModel, "_initialize_client", lambda self: None)
    model = KserveV2OcrModel(
        enabled=False,
        artifacts_path=None,
        options=_options(),
        accelerator_options=AcceleratorOptions(),
        enable_remote_services=False,
    )
    assert model._kserve_client is None
