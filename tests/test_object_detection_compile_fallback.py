"""Regression test: a failing torch.compile() must not fail the conversion."""

import pytest
import torch
import torch._dynamo

from docling.models.inference_engines.object_detection.transformers_engine import (
    TransformersObjectDetectionEngine,
)


class _CompileFailure(torch._dynamo.exc.TorchDynamoException):
    """Stands in for the backend failure raised on the first compiled forward."""


class _FailingCompiledModel(torch.nn.Module):
    def forward(self, **kwargs):
        raise _CompileFailure("No working C++ compiler found")


class _EagerModel(torch.nn.Module):
    def forward(self, **kwargs):
        return "eager-output"


class _BrokenModel(torch.nn.Module):
    def forward(self, **kwargs):
        raise ValueError("a genuine model error")


def _engine(model, uncompiled):
    engine = object.__new__(TransformersObjectDetectionEngine)
    engine._model = model
    engine._uncompiled_model = uncompiled
    return engine


def test_forward_falls_back_to_the_uncompiled_model():
    eager = _EagerModel()
    engine = _engine(_FailingCompiledModel(), eager)

    assert engine._forward(pixel_values=None) == "eager-output"
    # the swap is permanent, so later batches do not retry compilation
    assert engine._model is eager
    assert engine._uncompiled_model is None
    assert engine._forward(pixel_values=None) == "eager-output"


def test_forward_reraises_when_there_is_nothing_to_fall_back_to():
    engine = _engine(_FailingCompiledModel(), None)

    with pytest.raises(_CompileFailure):
        engine._forward(pixel_values=None)


def test_forward_does_not_mask_genuine_model_errors():
    engine = _engine(_BrokenModel(), _EagerModel())

    with pytest.raises(ValueError, match="a genuine model error"):
        engine._forward(pixel_values=None)
    # the uncompiled model is still held, since no fallback happened
    assert engine._uncompiled_model is not None
