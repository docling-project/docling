import importlib
import sys
import types

from openpyxl import Workbook


def _install_backend_import_stubs() -> None:
    docling_core_module = types.ModuleType("docling_core")
    docling_core_types_module = types.ModuleType("docling_core.types")
    docling_core_types_doc_module = types.ModuleType("docling_core.types.doc")

    for name in (
        "BoundingBox",
        "ContentLayer",
        "CoordOrigin",
        "DocItem",
        "DocItemLabel",
        "DoclingDocument",
        "DocumentOrigin",
        "GroupLabel",
        "ImageRef",
        "ProvenanceItem",
        "Size",
        "TableCell",
        "TableData",
    ):
        setattr(docling_core_types_doc_module, name, type(name, (), {}))

    docling_core_module.types = docling_core_types_module
    docling_core_types_module.doc = docling_core_types_doc_module

    abstract_backend_module = types.ModuleType("docling.backend.abstract_backend")
    abstract_backend_module.DeclarativeDocumentBackend = type(
        "DeclarativeDocumentBackend", (), {}
    )
    abstract_backend_module.PaginatedDocumentBackend = type(
        "PaginatedDocumentBackend", (), {}
    )

    base_models_module = types.ModuleType("docling.datamodel.base_models")
    base_models_module.InputFormat = type("InputFormat", (), {"XLSX": "xlsx"})

    document_module = types.ModuleType("docling.datamodel.document")
    document_module.InputDocument = type("InputDocument", (), {})

    sys.modules["docling_core"] = docling_core_module
    sys.modules["docling_core.types"] = docling_core_types_module
    sys.modules["docling_core.types.doc"] = docling_core_types_doc_module
    sys.modules["docling.backend.abstract_backend"] = abstract_backend_module
    sys.modules["docling.datamodel.base_models"] = base_models_module
    sys.modules["docling.datamodel.document"] = document_module


def _load_msexcel_backend():
    try:
        from docling.backend.msexcel_backend import MsExcelDocumentBackend
        from docling.datamodel.backend_options import MsExcelBackendOptions

        return MsExcelDocumentBackend, MsExcelBackendOptions
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "docling_core",
            "docling_core.types",
            "docling_core.types.doc",
        }:
            raise

        sys.modules.pop("docling.backend.msexcel_backend", None)
        _install_backend_import_stubs()

        backend_module = importlib.import_module("docling.backend.msexcel_backend")
        from docling.datamodel.backend_options import MsExcelBackendOptions

        return backend_module.MsExcelDocumentBackend, MsExcelBackendOptions


def test_find_data_tables_handles_a_filled_last_excel_row():
    MsExcelDocumentBackend, MsExcelBackendOptions = _load_msexcel_backend()

    workbook = Workbook()
    sheet = workbook.active
    sheet["A1048576"] = "last row"

    backend = object.__new__(MsExcelDocumentBackend)
    backend.options = MsExcelBackendOptions()

    tables = backend._find_data_tables(sheet)

    assert len(tables) == 1

    table = tables[0]
    assert table.anchor == (0, 1048575)
    assert table.num_rows == 1
    assert table.num_cols == 1
    assert len(table.data) == 1
    assert table.data[0].text == "last row"
