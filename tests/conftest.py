from pathlib import Path

import pytest

TEST_DATA_DIR = Path("./tests/data/")


@pytest.fixture
def test_data_directory() -> Path:
    return TEST_DATA_DIR
