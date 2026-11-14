# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Formula results show the 15 significant digits that Excel shows."""

import re
import zipfile
from pathlib import Path

from openpyxl import Workbook

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

# Label, formula, the result that Excel caches with 17 significant digits,
# number format, and the expected cell text.
_ROWS = [
    ("Total", "=SUM(B2:B3)", "30.299999999999997", "General", "30.3"),
    ("Sum", "=0.1+0.2", "0.30000000000000004", "General", "0.3"),
    ("Price with tax", "=1.3*1.15", "1.4949999999999999", '"$"#,##0.00', "$1.50"),
    # Rounding keeps significant digits, so a small result keeps its value.
    (
        "Residual",
        "=1-0.9-0.1",
        "-2.7755575615628914E-17",
        "General",
        "-2.77555756156289e-17",
    ),
]


def _write_workbook(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Item", "Amount"])
    sheet.append(["Coffee", 10.1])
    sheet.append(["Tea", 20.2])
    for row_index, (label, formula, _, number_format, _) in enumerate(_ROWS, start=4):
        sheet.cell(row_index, 1, label)
        sheet.cell(row_index, 2, formula).number_format = number_format
    workbook.save(path)

    # openpyxl writes formulas without a result. Add the results as Excel
    # stores them in the sheet XML.
    sheet_member = "xl/worksheets/sheet1.xml"
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    sheet_xml = members[sheet_member].decode("utf-8")
    for row_index, (_, _, cached, _, _) in enumerate(_ROWS, start=4):
        sheet_xml, count = re.subn(
            rf'(<c r="B{row_index}"[^>]*><f>[^<]*</f>)<v></v>',
            rf"\g<1><v>{cached}</v>",
            sheet_xml,
        )
        assert count == 1
    members[sheet_member] = sheet_xml.encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_xlsx_formula_results_keep_excel_precision(tmp_path: Path) -> None:
    path = tmp_path / "formula_results.xlsx"
    _write_workbook(path)

    converter = DocumentConverter(allowed_formats=[InputFormat.XLSX])
    table = converter.convert(path).document.tables[0].data
    amounts = [
        cell.text
        for cell in sorted(table.table_cells, key=lambda c: c.start_row_offset_idx)
        if cell.start_col_offset_idx == 1
    ]

    assert amounts == ["Amount", "10.1", "20.2", *[row[4] for row in _ROWS]]
