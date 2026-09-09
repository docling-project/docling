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
