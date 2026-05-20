"""
Unit tests for TabularLoader in facade/attachment_processor.py.

이슈 #51 회귀 방지 테스트.
첨부용 전처리기에 행 수가 많은 csv/xlsx를 업로드하면
weaviate context exceeded 에러가 발생하던 문제(원인: weaviate timeout)와
관련해, TabularLoader가 큰 tabular 파일을 받아도
vectors 생성 자체는 정상적으로 끝나는지 검증한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


def _import_tabular_loader():
    try:
        from facade.attachment_processor import TabularLoader, GenOSVectorMeta
        return TabularLoader, GenOSVectorMeta
    except ModuleNotFoundError:
        sys.path.append(str(Path(__file__).resolve().parents[3]))
        from facade.attachment_processor import TabularLoader, GenOSVectorMeta
        return TabularLoader, GenOSVectorMeta


def _make_large_dataframe(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({
        "id": list(range(n_rows)),
        "name": [f"row_{i}" for i in range(n_rows)],
        "value": [i * 1.5 for i in range(n_rows)],
        "flag": [(i % 2 == 0) for i in range(n_rows)],
    })


@pytest.fixture
def large_csv_path(tmp_path: Path) -> Path:
    df = _make_large_dataframe(5000)
    path = tmp_path / "large_sample.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture
def large_xlsx_path(tmp_path: Path) -> Path:
    pytest.importorskip("openpyxl")
    df = _make_large_dataframe(5000)
    path = tmp_path / "large_sample.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def multi_sheet_xlsx_path(tmp_path: Path) -> Path:
    pytest.importorskip("openpyxl")
    path = tmp_path / "multi_sheet.xlsx"
    with pd.ExcelWriter(path) as writer:
        _make_large_dataframe(2000).to_excel(writer, sheet_name="alpha", index=False)
        _make_large_dataframe(2000).to_excel(writer, sheet_name="beta", index=False)
    return path


@pytest.mark.unit
class TestTabularLoaderLargeInputs:
    """이슈 51: 큰 csv/xlsx 입력에 대해 vectors 생성이 정상 완료되는지 검증."""

    def test_large_csv_loads_without_error(self, large_csv_path: Path):
        TabularLoader, _ = _import_tabular_loader()

        loader = TabularLoader(str(large_csv_path), ".csv")

        assert loader.data_dict is not None
        assert "data" in loader.data_dict
        assert len(loader.data_dict["data"]) == 1
        assert len(loader.data_dict["data"][0]["data_rows"]) == 5000

    def test_large_csv_returns_vectors(self, large_csv_path: Path):
        TabularLoader, GenOSVectorMeta = _import_tabular_loader()

        loader = TabularLoader(str(large_csv_path), ".csv")
        vectors = loader.return_vectormeta_format()

        assert isinstance(vectors, list)
        assert len(vectors) >= 1
        v0 = vectors[0]
        assert isinstance(v0, GenOSVectorMeta)
        assert isinstance(v0.text, str)
        assert v0.text.startswith("[DA] ")
        assert len(v0.text) > 0

    def test_large_xlsx_loads_without_error(self, large_xlsx_path: Path):
        TabularLoader, _ = _import_tabular_loader()

        loader = TabularLoader(str(large_xlsx_path), ".xlsx")

        assert loader.data_dict is not None
        assert len(loader.data_dict["data"]) == 1
        assert len(loader.data_dict["data"][0]["data_rows"]) == 5000

    def test_large_xlsx_returns_vectors(self, large_xlsx_path: Path):
        TabularLoader, GenOSVectorMeta = _import_tabular_loader()

        loader = TabularLoader(str(large_xlsx_path), ".xlsx")
        vectors = loader.return_vectormeta_format()

        assert isinstance(vectors, list)
        assert len(vectors) >= 1
        v0 = vectors[0]
        assert isinstance(v0, GenOSVectorMeta)
        assert isinstance(v0.text, str)
        assert v0.text.startswith("[DA] ")

    def test_multi_sheet_xlsx_includes_all_sheets(self, multi_sheet_xlsx_path: Path):
        TabularLoader, _ = _import_tabular_loader()

        loader = TabularLoader(str(multi_sheet_xlsx_path), ".xlsx")

        sheet_names = [d["sheet_name"] for d in loader.data_dict["data"]]
        assert sheet_names == ["alpha", "beta"]
        for sheet in loader.data_dict["data"]:
            assert len(sheet["data_rows"]) == 2000

    def test_large_csv_vector_contains_all_row_ids(self, large_csv_path: Path):
        """모든 행이 vector text에 포함되는지(데이터 손실 없음) 확인."""
        TabularLoader, _ = _import_tabular_loader()

        loader = TabularLoader(str(large_csv_path), ".csv")
        vectors = loader.return_vectormeta_format()

        joined_text = "".join(v.text for v in vectors)
        assert "row_0" in joined_text
        assert "row_4999" in joined_text
