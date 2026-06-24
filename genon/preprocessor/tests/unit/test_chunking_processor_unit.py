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


# ----------------------------------------------------------------------
# parse-format(비-docling) 공통 청킹 — parser 가 docling 을 못 만드는 포맷
# (audio, csv/xlsx, ppt/pptx/doc, txt/json/md, 이미지) 연동.
# 포맷은 file_path 확장자가 아니라 payload(element) 내용으로 판별한다.
# ----------------------------------------------------------------------

def test_classify_payload_shapes():
    """payload 형태 판별: docling/parse-format/envelope/garbage."""
    cp = pytest.importorskip("facade.chunking_processor")

    assert cp._classify_payload({"document": {"x": 1}}) == ("docling", {"x": 1})
    assert cp._classify_payload({"elements": [{"content": "a"}]}) == ("parse", [{"content": "a"}])
    # docling 우선: parser docling 응답은 _normalize_response 로 빈 elements 도 함께 가질 수 있음
    assert cp._classify_payload({"document": {"x": 1}, "elements": []})[0] == "docling"
    # envelope
    assert cp._classify_payload({"code": 0, "data": {"elements": [{"content": "a"}]}})[0] == "parse"
    # raw docling dict
    assert cp._classify_payload({"schema_name": "DoclingDocument", "body": {}})[0] == "docling"
    with pytest.raises(cp.GenosServiceException):
        cp._classify_payload({"unknown": 1})


def test_chunker_parse_format_audio_single_vector():
    """audio parse-format([AUDIO] 접두사) → 단일 벡터(분할 없음)."""
    cp = pytest.importorskip("facade.chunking_processor")

    transcript = "[AUDIO] 안녕하세요 모니모 음성 안내입니다. " * 50
    elements = [{"category": "paragraph", "content": transcript, "coordinates": [], "id": 0, "page": 1}]

    chunker = cp.DocumentProcessor()
    vectors = asyncio.run(
        chunker(request=None, file_path="/data/voice.json", document={"elements": elements})
    )

    assert len(vectors) == 1
    assert vectors[0].text.startswith("[AUDIO]")


def test_chunker_parse_format_tabular_da_vector():
    """csv/xlsx parse-format(category=='table' 전부) → 단일 [DA] 벡터."""
    cp = pytest.importorskip("facade.chunking_processor")

    elements = [
        {"category": "table", "content": "<table><tr><td>a</td></tr></table>", "page": 1, "id": 0},
        {"category": "table", "content": "<table><tr><td>b</td></tr></table>", "page": 2, "id": 1},
    ]

    chunker = cp.DocumentProcessor()
    vectors = asyncio.run(
        chunker(request=None, file_path="/data/sheet.json", document={"elements": elements})
    )

    assert len(vectors) == 1
    assert vectors[0].text.startswith("[DA] ")


def test_chunker_parse_format_text_multi_chunk():
    """텍스트 parse-format → RecursiveCharacterTextSplitter 다중 청킹, 인덱스 연속."""
    cp = pytest.importorskip("facade.chunking_processor")

    elements = [
        {"category": "paragraph", "content": "가나다라마바사아자차카타파하 " * 20, "page": 1, "id": 0},
        {"category": "paragraph", "content": "ABCDEFG HIJKLMN OPQRSTU " * 20, "page": 2, "id": 1},
    ]

    chunker = cp.DocumentProcessor()
    vectors = asyncio.run(
        chunker(
            request=None, file_path="/data/note.json",
            document={"elements": elements}, chunk_size=50, chunk_overlap=0,
        )
    )

    assert len(vectors) >= 2
    # i_chunk_on_doc 가 0..N-1 연속
    assert [v.i_chunk_on_doc for v in vectors] == list(range(len(vectors)))
    # page 메타가 1-based 로 보존(parser element page 그대로, +1 하지 않음)
    assert set(v.i_page for v in vectors) <= {1, 2}
