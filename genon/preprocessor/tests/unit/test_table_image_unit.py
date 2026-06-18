"""
table_image 기능 단위 테스트.

대상: intelligent_processor / convert_processor 의 DocumentProcessor 가 동일하게 동작하는지.
- 설정 파싱: table_image.enable → self.table_image_enabled, enable 시 generate_page_images 강제 True.
- 메타: GenOSVectorMetaBuilder.set_media_files / DocumentProcessor.get_media_files 의 include_tables.
- 저장: DocumentProcessor._save_table_images 가 표를 PNG 로 저장하고 item.image.uri 를 파일경로로 설정.

의존성(docling 등) 미가용 환경에서는 importorskip 으로 자동 skip 된다(CI gate).
"""
import shutil
from pathlib import Path

import pytest
import yaml

# docling_core 가 없으면 전체 모듈 skip (facade 도 동일 의존성)
pytest.importorskip("docling_core")

from docling_core.types.doc import (  # noqa: E402
    BoundingBox,
    DoclingDocument,
    ImageRef,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.doc.base import CoordOrigin  # noqa: E402
from PIL import Image  # noqa: E402

_MODULES = ["intelligent_processor", "convert_processor"]

_DEFAULT_CONFIG = {
    "intelligent_processor": "intelligent_processor_config.yaml",
    "convert_processor": "convert_processor_config.yaml",
}


# ─── helpers ────────────────────────────────────────────────────────────────

def _load_mod(module_name: str):
    return pytest.importorskip(f"facade.{module_name}")


def _make_config(tmp_path: Path, module_name: str, *, enable, generate_page_images=None) -> str:
    """출고 config 를 복사하고 table_image.enable(+선택적 generate_page_images)만 덮어쓴 임시 config.

    enrichment 의 prompt 파일은 config 디렉터리 기준 상대경로로 resolve 되므로,
    resource 디렉터리의 보조 파일(prompt_*.md 등)을 tmp_path 로 함께 복사한다.
    """
    resource_dir = Path(__file__).resolve().parents[2] / "resource"
    for f in resource_dir.iterdir():
        if f.is_file() and f.name != _DEFAULT_CONFIG[module_name]:
            shutil.copy(f, tmp_path / f.name)

    cfg = yaml.safe_load((resource_dir / _DEFAULT_CONFIG[module_name]).read_text(encoding="utf-8"))
    if enable is None:
        cfg.pop("table_image", None)
    else:
        cfg["table_image"] = {"enable": enable}
    if generate_page_images is not None:
        cfg.setdefault("pdf_pipeline", {})["generate_page_images"] = generate_page_images
    out = tmp_path / _DEFAULT_CONFIG[module_name]
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(out)


def _init_processor(module_name: str, config_path: str):
    DocumentProcessor = _load_mod(module_name).DocumentProcessor
    try:
        return DocumentProcessor(config_path=config_path)
    except Exception as e:  # noqa: BLE001 - 모델/네트워크 등 환경 의존
        pytest.skip(f"DocumentProcessor init unavailable: {e}")


def _build_doc_with_table():
    """페이지 이미지 + 표(bbox) 를 가진 최소 DoclingDocument 와 그 TableItem 반환."""
    doc = DoclingDocument(name="t")
    page_img = Image.new("RGB", (400, 400), (255, 255, 255))
    for x in range(50, 200):
        for y in range(50, 200):
            page_img.putpixel((x, y), (255, 0, 0))  # 표 영역 구분용
    doc.add_page(page_no=1, size=Size(width=400, height=400),
                 image=ImageRef.from_pil(page_img, dpi=72))
    tdata = TableData(num_rows=1, num_cols=1, table_cells=[
        TableCell(text="A", row_span=1, col_span=1, start_row_offset_idx=0,
                  end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1)])
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox(l=50, t=50, r=200, b=200, coord_origin=CoordOrigin.TOPLEFT),
        charspan=(0, 1))
    table = doc.add_table(data=tdata, prov=prov)
    return doc, table


# ─── 설정 파싱 ────────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
def test_table_image_enable_true(tmp_path, module_name):
    proc = _init_processor(module_name, _make_config(tmp_path, module_name, enable=True))
    assert proc.table_image_enabled is True


@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
def test_table_image_enable_false(tmp_path, module_name):
    proc = _init_processor(module_name, _make_config(tmp_path, module_name, enable=False))
    assert proc.table_image_enabled is False


@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
def test_table_image_absent_defaults_false(tmp_path, module_name):
    """table_image 섹션 자체가 없어도 기본 False (하위 호환)."""
    proc = _init_processor(module_name, _make_config(tmp_path, module_name, enable=None))
    assert proc.table_image_enabled is False


@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
def test_enable_forces_generate_page_images(tmp_path, module_name):
    """enable=true 면 generate_page_images=false 설정이어도 True 로 강제 보정된다."""
    proc = _init_processor(
        module_name,
        _make_config(tmp_path, module_name, enable=True, generate_page_images=False),
    )
    assert proc.pipe_line_options.generate_page_images is True


@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
def test_shipped_config_default_is_false(module_name):
    """출고 resource/resource_dev/resource_product 의 table_image.enable 이 모두 false."""
    base = Path(__file__).resolve().parents[2]
    for sub in ("resource", "resource_dev", "resource_product"):
        path = base / sub / _DEFAULT_CONFIG[module_name]
        if not path.exists():
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        ti = cfg.get("table_image")
        assert ti is not None, f"{path} 에 table_image 섹션 누락"
        assert ti.get("enable") is False, f"{path} table_image.enable 이 false 가 아님: {ti}"
        assert "answer_strategy" not in ti, f"{path} 에 제거된 answer_strategy 잔존"


# ─── 메타 / 저장 ──────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
def test_set_media_files_skips_table_without_image(module_name):
    """이미지가 아직 없는(저장 전) 표는 include_tables=True 여도 수집되지 않아야 함."""
    import json
    mod = _load_mod(module_name)
    _doc, table = _build_doc_with_table()  # _save_table_images 호출 전 → table.image is None
    media = json.loads(mod.GenOSVectorMetaBuilder().set_media_files([table], include_tables=True).media_files)
    assert media == []


@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
def test_save_table_images_and_media_files(tmp_path, module_name):
    import json
    mod = _load_mod(module_name)
    doc, table = _build_doc_with_table()
    image_dir = tmp_path / "doc1"

    # _save_table_images 는 self 를 사용하지 않으므로 None 으로 호출 가능
    mod.DocumentProcessor._save_table_images(None, doc, image_dir=image_dir, reference_path=tmp_path)

    # 1) PNG 저장 + uri 가 base64 가 아닌 파일 경로
    files = list(image_dir.glob("table_*.png"))
    assert files, "table_*.png 가 저장되어야 함"
    uri = str(table.image.uri)
    assert not uri.startswith("data:"), "uri 가 base64 data URI 면 안 됨"
    assert uri.endswith(".png") and "table_" in uri

    # 2) set_media_files(include_tables=True) → type='table_image', ref 일치
    media = json.loads(mod.GenOSVectorMetaBuilder().set_media_files([table], include_tables=True).media_files)
    assert len(media) == 1
    assert media[0]["type"] == "table_image"
    assert media[0]["ref"] == table.self_ref
    assert media[0]["name"].startswith("table_")

    # 3) include_tables=False(기본) → 표 미수집 (회귀 가드)
    media_off = json.loads(mod.GenOSVectorMetaBuilder().set_media_files([table]).media_files)
    assert media_off == []

    # 4) get_media_files(include_tables=True) → 업로드 목록에 표 PNG 포함
    uploads = mod.DocumentProcessor.get_media_files(None, [table], include_tables=True)
    assert len(uploads) == 1
    assert uploads[0]["name"].startswith("table_") and uploads[0]["path"].endswith(".png")

    # 5) chunk_bboxes 의 table ref 와 media ref 가 동일 → 조인 가능
    cb = json.loads(mod.GenOSVectorMetaBuilder().set_chunk_bboxes([table], doc).chunk_bboxes)
    assert cb and cb[0]["ref"] == media[0]["ref"]
