"""Regression tests for the PII obfuscation example's output handling."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "docs/examples/pii_obfuscate.py"


@pytest.mark.parametrize("save_original", [False, True])
def test_pii_example_does_not_expose_values_by_default(
    tmp_path, monkeypatch, caplog, save_original
):
    spec = importlib.util.spec_from_file_location("pii_obfuscate_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    synthetic_name = "Demo Person"

    class FakeTextItem:
        def __init__(self, text):
            self.text = text
            self.orig = text

    class FakeTableItem:
        def __init__(self, text):
            self.data = SimpleNamespace(table_cells=[SimpleNamespace(text=text)])

    class FakeDocument:
        def __init__(self):
            self.text_item = FakeTextItem(f"Body contains {synthetic_name}")
            self.table_item = FakeTableItem(f"Table contains {synthetic_name}")

        def iterate_items(self):
            yield self.text_item, 0
            yield self.table_item, 0

        def save_as_markdown(self, path, **kwargs):
            content = (
                f"{self.text_item.text}\n{self.table_item.data.table_cells[0].text}"
            )
            Path(path).write_text(content, encoding="utf-8")

    document = FakeDocument()

    class FakeConverter:
        def __init__(self, **kwargs):
            pass

        def convert(self, *args, **kwargs):
            return SimpleNamespace(
                document=document,
                input=SimpleNamespace(file=Path("fixture.pdf")),
            )

    def fake_ner(text):
        start = text.find(synthetic_name)
        if start < 0:
            return []
        return [
            {
                "entity_group": "PER",
                "start": start,
                "end": start + len(synthetic_name),
                "word": synthetic_name,
            }
        ]

    monkeypatch.setattr(module, "DocumentConverter", FakeConverter)
    monkeypatch.setattr(module, "TextItem", FakeTextItem)
    monkeypatch.setattr(module, "TableItem", FakeTableItem)
    monkeypatch.setattr(module, "_build_simple_ner_pipeline", lambda: fake_ner)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pii_obfuscate.py"] + (["--save-original"] if save_original else []),
    )

    with caplog.at_level(logging.INFO):
        module.main()

    safe_file = tmp_path / "scratch" / "fixture.pdf-with-images-pii-obfuscated.md"
    original_file = tmp_path / "scratch" / "fixture.pdf-with-images-orig.md"

    assert safe_file.exists()
    assert synthetic_name not in safe_file.read_text()
    assert "person-1" in safe_file.read_text()

    assert original_file.exists() is save_original
    if save_original:
        assert synthetic_name in original_file.read_text()

    assert synthetic_name not in document.text_item.orig
    assert synthetic_name not in caplog.text
    assert "Obfuscated 1 unique entities" in caplog.text
