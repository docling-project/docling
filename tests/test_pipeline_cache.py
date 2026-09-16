# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    PictureDescriptionVlmOptions,
    PipelineOptions,
)
from docling.utils.pipeline_cache import create_pipeline_options_hash


def test_hash_is_stable():
    assert create_pipeline_options_hash(
        PdfPipelineOptions()
    ) == create_pipeline_options_hash(PdfPipelineOptions())


def test_scalar_change_changes_hash():
    a = PdfPipelineOptions()
    b = PdfPipelineOptions()
    b.document_timeout = (a.document_timeout or 0) + 1
    assert create_pipeline_options_hash(a) != create_pipeline_options_hash(b)


def test_subtype_in_base_typed_field_does_not_collide():
    # picture_description_options is typed as the base class; the default
    # dump truncates subtype fields, so an API and a VLM backend would share a
    # cache key. serialize_as_any keeps the distinguishing fields.
    api = PdfPipelineOptions()
    api.picture_description_options = PictureDescriptionApiOptions(url="http://x")
    vlm = PdfPipelineOptions()
    vlm.picture_description_options = PictureDescriptionVlmOptions(repo_id="r")
    assert create_pipeline_options_hash(api) != create_pipeline_options_hash(vlm)


def test_different_options_classes_do_not_collide():
    assert create_pipeline_options_hash(
        PipelineOptions()
    ) != create_pipeline_options_hash(PdfPipelineOptions())


def test_default_pdf_pipeline_options_hash_validity():
    h = create_pipeline_options_hash(PdfPipelineOptions())
    assert isinstance(h, str)
    assert len(h) == 32
    assert all(c in "0123456789abcdef" for c in h)


def test_fallback_when_serialize_as_any_raises_circular_reference(monkeypatch):
    from pydantic_core import PydanticSerializationError

    orig_dump = PdfPipelineOptions.model_dump_json

    def mock_dump_json(self, *args, **kwargs):
        if kwargs.get("serialize_as_any"):
            raise PydanticSerializationError(
                "ValueError: Circular reference detected (id repeated)"
            )
        return orig_dump(self, *args, **kwargs)

    monkeypatch.setattr(PdfPipelineOptions, "model_dump_json", mock_dump_json)

    # Asserts that default PdfPipelineOptions succeeds without raising
    h1 = create_pipeline_options_hash(PdfPipelineOptions())
    h2 = create_pipeline_options_hash(PdfPipelineOptions())
    assert h1 == h2
    assert len(h1) == 32

    # Asserts scalar changes alter the hash under fallback
    a = PdfPipelineOptions()
    b = PdfPipelineOptions()
    b.document_timeout = (a.document_timeout or 0) + 1
    assert create_pipeline_options_hash(a) != create_pipeline_options_hash(b)

    # Asserts subtype in base-typed field does not collide under fallback
    api = PdfPipelineOptions()
    api.picture_description_options = PictureDescriptionApiOptions(url="http://x")
    vlm = PdfPipelineOptions()
    vlm.picture_description_options = PictureDescriptionVlmOptions(repo_id="r")
    assert create_pipeline_options_hash(api) != create_pipeline_options_hash(vlm)

    # Asserts different classes do not collide under fallback
    assert create_pipeline_options_hash(
        PipelineOptions()
    ) != create_pipeline_options_hash(PdfPipelineOptions())


def test_fallback_resilience_when_sub_dump_fails(monkeypatch):
    from docling.utils.pipeline_cache import _dump_pipeline_options_fallback

    opts = PdfPipelineOptions()

    def fail_model_dump(self, *args, **kwargs):
        raise RuntimeError("Model dump failure")

    # 1. Test when base model_dump raises
    orig_base_dump = PdfPipelineOptions.model_dump
    monkeypatch.setattr(PdfPipelineOptions, "model_dump", fail_model_dump)
    dump_str = _dump_pipeline_options_fallback(opts)
    assert '"base": {}' in dump_str
    monkeypatch.setattr(PdfPipelineOptions, "model_dump", orig_base_dump)

    # 2. Test when a nested sub-model model_dump raises
    sub_class = type(opts.picture_description_options)
    monkeypatch.setattr(sub_class, "model_dump", fail_model_dump)
    dump_str_sub = _dump_pipeline_options_fallback(opts)
    assert sub_class.__qualname__ in dump_str_sub
