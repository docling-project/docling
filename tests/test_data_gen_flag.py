import os

from pydantic import TypeAdapter

GEN_TEST_DATA = TypeAdapter(bool).validate_python(os.getenv("DOCLING_GEN_TEST_DATA", 0))
IS_CI = TypeAdapter(bool).validate_python(os.getenv("CI", 0))


def test_gen_test_data_flag():
    assert not GEN_TEST_DATA
