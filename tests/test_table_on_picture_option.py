"""Unit tests for the ``try_table_on_picture`` table-detection fallback (#3410).

These cover the policy in isolation (which layout regions the table backends
consider, and when a prediction is kept) so the behaviour is verified without
downloading or running the table model.
"""

from docling_core.types.doc import DocItemLabel

from docling.datamodel.pipeline_options import (
    GraniteVisionTableStructureOptions,
    TableStructureOptions,
    TableStructureV2Options,
)
from docling.models.base_table_model import is_table_like, table_candidate_labels


def test_try_table_on_picture_is_off_by_default_on_every_backend():
    assert TableStructureOptions().try_table_on_picture is False
    assert TableStructureV2Options().try_table_on_picture is False
    assert GraniteVisionTableStructureOptions().try_table_on_picture is False


def test_enabling_the_flag_is_preserved():
    assert TableStructureOptions(try_table_on_picture=True).try_table_on_picture is True


def test_candidate_labels_exclude_pictures_by_default():
    labels = table_candidate_labels(try_table_on_picture=False)
    assert DocItemLabel.TABLE in labels
    assert DocItemLabel.DOCUMENT_INDEX in labels
    assert DocItemLabel.PICTURE not in labels


def test_candidate_labels_add_pictures_without_dropping_tables():
    labels = table_candidate_labels(try_table_on_picture=True)
    assert DocItemLabel.PICTURE in labels
    assert DocItemLabel.TABLE in labels
    assert DocItemLabel.DOCUMENT_INDEX in labels


def test_is_table_like_requires_at_least_two_rows_and_columns():
    assert is_table_like(2, 2) is True
    assert is_table_like(6, 3) is True
    assert is_table_like(1, 5) is False
    assert is_table_like(5, 1) is False
    assert is_table_like(0, 0) is False
