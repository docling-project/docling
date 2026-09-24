# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

from typer.testing import CliRunner

from docling.cli.main import app

runner = CliRunner()
FORM_PDF = Path("tests/data/pdf/sources/acroform_sample.pdf")


def test_form_fields_rejected_with_doctags_before_converting(tmp_path: Path) -> None:
    # DocTags has no tokens for form fields and its export raises on them, so
    # the combination is refused at argument time, before any model loads.
    result = runner.invoke(
        app,
        [
            str(FORM_PDF),
            "--extract-form-fields",
            "--to",
            "doctags",
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "--extract-form-fields" in result.output
    assert "doctags" in result.output
    assert list(tmp_path.iterdir()) == []
