"""
모니모 Parsing(docling) + Chunk API 단위 테스트 (#283 / #284).

실제 호출 기반(mock 금지). 의존성(docling 등) 미가용 환경에서는 importorskip 으로 자동 skip(CI gate).
- #283: parser_processor 의 output.format="docling" 이 복원 가능한 DoclingDocument JSON 을 반환하는지
        (DoclingDocument.model_validate 무손실 round-trip).
- #284: chunking_processor 가 그 docling JSON 을 입력받아 GenOSVectorMeta 리스트를 반환하는지.
"""
import asyncio

import pytest

from docling_core.types import DoclingDocument
from docling_core.types.doc import DocItemLabel


def _build_doc() -> DoclingDocument:
    doc = DoclingDocument(name="monimo_sample")
    doc.add_title(text="모니모 약관")
    h1 = doc.add_heading(text="제1조 목적", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="이 약관은 모니모 서비스 이용에 관한 사항을 규정한다.", parent=h1)
    doc.add_text(label=DocItemLabel.TEXT, text="회사는 본 약관을 서비스 화면에 게시한다.", parent=h1)
    h2 = doc.add_heading(text="제2조 정의", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="'회원'이란 약관에 동의하고 가입한 자를 말한다.", parent=h2)
    return doc


def test_parser_docling_output_roundtrip():
    """#283: output.format='docling' → data.document 가 무손실 복원 가능해야 한다."""
    pp = pytest.importorskip("facade.parser_processor")

    parser = object.__new__(pp.DocumentProcessor)  # __init__(네트워크/config) 우회
    parser._output_format = "docling"
    parser._table_format = "html"

    doc = _build_doc()
    resp = parser._build_docling_response(doc)

    assert "document" in resp
    assert resp["usage"]["pages"] == doc.num_pages()

    restored = DoclingDocument.model_validate(resp["document"])
    assert [t.text for t in restored.texts] == [t.text for t in doc.texts]


def test_chunker_consumes_docling_json():
    """#284: chunking_processor 가 docling JSON 을 입력받아 GenOSVectorMeta 리스트 반환."""
    cp = pytest.importorskip("facade.chunking_processor")

    assert getattr(cp.DocumentProcessor, "IS_CHUNKER", False) is True

    doc_dict = _build_doc().model_dump(mode="json")  # parser output.format='docling' 직렬화와 동일 경로

    chunker = cp.DocumentProcessor()
    vectors = asyncio.run(
        chunker(request=None, file_path="/data/monimo_sample.pdf", document=doc_dict)
    )

    assert isinstance(vectors, list) and len(vectors) >= 1
    v0 = vectors[0]
    for field in ("text", "n_char", "i_page", "i_chunk_on_doc", "n_chunk_of_doc"):
        assert hasattr(v0, field), field
    # 마지막 청크 인덱스 == 전체 청크 수 - 1
    assert vectors[-1].i_chunk_on_doc == len(vectors) - 1


def test_chunker_missing_document_raises():
    """#284: document 입력이 없으면 GenosServiceException."""
    cp = pytest.importorskip("facade.chunking_processor")

    chunker = cp.DocumentProcessor()
    with pytest.raises(cp.GenosServiceException):
        asyncio.run(chunker(request=None, file_path=""))
