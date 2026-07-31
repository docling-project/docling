"""Tests for the package-level ``docling.__version__`` attribute."""

import importlib.metadata

import docling


def test_version_is_exposed():
    assert isinstance(docling.__version__, str)
    assert docling.__version__


def test_version_matches_installed_distribution():
    for name in ("docling", "docling-slim"):
        try:
            expected = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        assert docling.__version__ == expected
        return
    assert docling.__version__ == "unknown"


def test_resolve_version_falls_back_to_unknown(monkeypatch):
    def _missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    assert docling._resolve_version() == "unknown"
