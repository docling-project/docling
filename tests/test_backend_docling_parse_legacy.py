from pathlib import Path

import pytest

from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

INPUT_FILE = "./tests/data/pdf/2206.01062.pdf"


@pytest.mark.parametrize(
    "cls",
    [DoclingParseV2DocumentBackend, DoclingParseV4DocumentBackend],
)
def test_emits_future_warning(cls):
    with pytest.warns(FutureWarning, match="DoclingParse"):
        InputDocument(
            path_or_stream=Path(INPUT_FILE), format=InputFormat.PDF, backend=cls
        )
