from pathlib import Path

import pytest

from ..test_data_gen_flag import GEN_TEST_DATA

GENERATE = GEN_TEST_DATA
LATEX_DATA_DIR = Path("./tests/data/latex/")


@pytest.fixture(scope="module")
def latex_paths() -> list[Path]:
    """Find all LaTeX files in the test data directory."""
    directory = Path("./tests/data/latex/")
    if not directory.exists():
        return []

    paths = list(directory.glob("*.tex"))

    for subdir in directory.iterdir():
        if subdir.is_dir():
            if (subdir / "main.tex").exists():
                paths.append(subdir / "main.tex")
            elif (subdir / f"arxiv_{subdir.name}.tex").exists():
                paths.append(subdir / f"arxiv_{subdir.name}.tex")

    return sorted(paths)
