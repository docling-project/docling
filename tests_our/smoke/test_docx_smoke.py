from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "sample_files"
DOCX_SAMPLES = sorted(SAMPLE_DIR.glob("*.docx"))


@pytest.mark.smoke
@pytest.mark.skipif(len(DOCX_SAMPLES) == 0, reason="no .docx samples found")
@pytest.mark.parametrize("sample", DOCX_SAMPLES, ids=lambda p: p.name)
def test_docx_smoke(basic_processor, sample: Path):
    dp = basic_processor()

    doc = dp.load_documents(str(sample))
    assert doc is not None
    if hasattr(doc, "num_pages"):
        assert doc.num_pages() >= 1

    chunks = dp.split_documents(doc)
    assert isinstance(chunks, list) and len(chunks) >= 1


@pytest.mark.smoke
@pytest.mark.skipif(len(DOCX_SAMPLES) == 0, reason="no .docx samples found")
@pytest.mark.parametrize("sample", DOCX_SAMPLES, ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_vector_schema_docx(basic_processor, sample: Path):
    dp = basic_processor()

    vectors = await dp(None, str(sample))
    assert isinstance(vectors, list) and len(vectors) >= 1
    v = vectors[0]
    if hasattr(v, "model_dump"):
        v = v.model_dump()
    required = [
        "text",
        "n_char",
        "n_word",
        "n_line",
        "i_page",
        "i_chunk_on_page",
        "i_chunk_on_doc",
    ]
    for k in required:
        assert k in v
    assert isinstance(v["text"], str)
    for k in [x for x in required if x != "text"]:
        assert isinstance(v[k], int)




