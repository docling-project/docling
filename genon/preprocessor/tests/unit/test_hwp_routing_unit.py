"""
HWP/HWPX 라우팅 및 백엔드 선택 로직에 대한 단위 테스트.
실제 파일 I/O 없이 mock으로 격리 검증한다.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# 1. 확장자 → HwpProcessor 라우팅 검증
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("ext", [".hwp", ".hwpx", ".hml"])
async def test_hwp_extension_routes_to_hwp_processor(attachment_processor, ext):
    """
    .hwp/.hwpx/.hml 확장자는 DocumentProcessor 내부에서
    반드시 self.hwp_processor 로 위임되어야 한다 (.hml 은 이슈 #323).
    """
    dp = attachment_processor()

    # hwp_processor를 mock으로 교체 — 실제 파싱 없이 호출 여부만 확인
    dp.hwp_processor = AsyncMock(return_value=[{"text": "mock"}])

    fake_request = MagicMock()
    fake_request.is_disconnected = AsyncMock(return_value=False)

    await dp(fake_request, f"/tmp/test_file{ext}")

    dp.hwp_processor.assert_called_once()
    # 첫 번째 위치 인자에 file_path가 포함되어야 함
    call_args = dp.hwp_processor.call_args
    assert ext in call_args.args[1] or ext in str(call_args)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("ext", [".pdf", ".docx", ".pptx"])
async def test_non_hwp_extension_does_not_route_to_hwp_processor(attachment_processor, ext):
    """
    HWP/HWPX가 아닌 확장자는 hwp_processor로 가면 안 된다.
    """
    dp = attachment_processor()
    dp.hwp_processor = AsyncMock(return_value=[])
    dp.docx_processor = AsyncMock(return_value=[])
    dp.load_documents = MagicMock(return_value=[])
    dp.split_documents = MagicMock(return_value=[])
    dp.compose_vectors = MagicMock(return_value=[])

    fake_request = MagicMock()
    fake_request.is_disconnected = AsyncMock(return_value=False)

    await dp(fake_request, f"/tmp/test_file{ext}")

    dp.hwp_processor.assert_not_called()


# ---------------------------------------------------------------------------
# 2. use_hwp_sdk 플래그 → 백엔드 선택 검증
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("use_hwp_sdk, expected_backend", [
    (True, "GenosHwpDocumentBackend"),
    (False, "HwpDocumentBackend"),
])
def test_use_hwp_sdk_flag_selects_correct_hwp_backend(use_hwp_sdk, expected_backend):
    """
    use_hwp_sdk=True  → GenosHwpDocumentBackend (hwp_sdk)
    use_hwp_sdk=False → HwpDocumentBackend (레거시 XML 변환)
    InputFormat.HWP 키 기준으로 확인.
    """
    from facade.attachment_processor import HwpProcessor
    from docling.datamodel.base_models import InputFormat

    with patch("facade.attachment_processor.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = MagicMock(
            document=MagicMock()
        )
        proc = HwpProcessor()
        proc.load_documents("/tmp/fake.hwp", use_hwp_sdk=use_hwp_sdk)

        call_kwargs = MockConverter.call_args[1]
        backend_cls = call_kwargs["format_options"][InputFormat.HWP].backend
        assert backend_cls.__name__ == expected_backend


@pytest.mark.unit
@pytest.mark.parametrize("use_hwp_sdk, expected_backend", [
    (True, "GenosHwpDocumentBackend"),
    (False, "HwpxDocumentBackend"),
])
def test_use_hwp_sdk_flag_selects_correct_hwpx_backend(use_hwp_sdk, expected_backend):
    """
    HWPX에 대해서도 use_hwp_sdk 플래그에 따른 백엔드 선택 확인.
    InputFormat.XML_HWPX 키 기준.
    """
    from facade.attachment_processor import HwpProcessor
    from docling.datamodel.base_models import InputFormat

    with patch("facade.attachment_processor.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = MagicMock(
            document=MagicMock()
        )
        proc = HwpProcessor()
        proc.load_documents("/tmp/fake.hwpx", use_hwp_sdk=use_hwp_sdk)

        call_kwargs = MockConverter.call_args[1]
        backend_cls = call_kwargs["format_options"][InputFormat.XML_HWPX].backend
        assert backend_cls.__name__ == expected_backend


# ---------------------------------------------------------------------------
# 3. dump_sdk_output 옵션 전파 검증
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_dump_sdk_output_disabled_when_sdk_off():
    """
    use_hwp_sdk=False일 때 dump_sdk_output=True를 줘도
    pipeline_options.dump_sdk_output은 False여야 한다.
    PipelineOptions는 Pydantic 모델이므로 mock 없이 실제 인스턴스를 사용하고,
    DocumentConverter에 전달된 값만 검증한다.
    """
    from facade.attachment_processor import HwpProcessor
    from docling.datamodel.base_models import InputFormat

    with patch("facade.attachment_processor.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = MagicMock(
            document=MagicMock()
        )
        proc = HwpProcessor()
        proc.load_documents(
            "/tmp/fake.hwp",
            use_hwp_sdk=False,
            dump_sdk_output=True,
        )

        call_kwargs = MockConverter.call_args[1]
        hwp_option = call_kwargs["format_options"][InputFormat.HWP]
        assert hwp_option.pipeline_options.dump_sdk_output is False


# ---------------------------------------------------------------------------
# 4. .hml(HWPML) 포맷 감지/추론 검증 (이슈 #323, hwp_sdk 260713+)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hml_extension_maps_to_xml_hwpx_format():
    """hml 확장자는 XML_HWPX 포맷으로 매핑되어 GenosHwpDocumentBackend 로 라우팅된다."""
    from docling.datamodel.base_models import FormatToExtensions, InputFormat

    assert "hml" in FormatToExtensions[InputFormat.XML_HWPX]


@pytest.mark.unit
def test_hml_path_guesses_xml_hwpx_format(tmp_path):
    """실제 .hml(HWPML XML) 파일이 _guess_format 에서 XML_HWPX 로 감지된다."""
    from docling.datamodel.document import _DocumentConversionInput
    from docling.datamodel.base_models import InputFormat

    hml = tmp_path / "sample.hml"
    hml.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<HWPML Version="2.8"></HWPML>',
        encoding="utf-8",
    )
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[])
    assert conv_input._guess_format(hml) == InputFormat.XML_HWPX


@pytest.mark.unit
@pytest.mark.parametrize("header, expected", [
    (b"PK\x03\x04....", ".hwpx"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", ".hwp"),
    (b'<?xml version="1.0"?><HWPML Version="2.8">', ".hml"),
])
def test_infer_suffix_from_stream_magic(header, expected):
    """BytesIO 입력 시 매직 바이트로 확장자를 추론한다 (.hml 은 HWPML 루트 태그)."""
    from io import BytesIO
    from unittest.mock import MagicMock
    from docling.backend.genos_hwp_backend import GenosHwpDocumentBackend

    in_doc = MagicMock()
    in_doc.file = None
    in_doc.format = None
    assert GenosHwpDocumentBackend._infer_suffix(BytesIO(header), in_doc) == expected
