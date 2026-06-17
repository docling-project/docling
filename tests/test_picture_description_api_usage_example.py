import importlib.util
from pathlib import Path


def _load_example_module():
    module_path = (
        Path(__file__).parents[1]
        / "docs"
        / "examples"
        / "picture_description_api_usage.py"
    )
    spec = importlib.util.spec_from_file_location(
        "picture_description_api_usage", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_api_url_uses_azure_endpoint_and_header(monkeypatch):
    monkeypatch.setenv("AZURE_API_BASE", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment-name")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    monkeypatch.setenv("PICTURE_DESCRIPTION_API_URL", "")
    monkeypatch.setenv("PICTURE_DESCRIPTION_MODEL", "ignored-for-azure")

    example = _load_example_module()

    options = example._build_picture_description_options()

    assert str(options.url) == (
        "https://example.openai.azure.com/openai/deployments/deployment-name/"
        "chat/completions?api-version=2025-01-01-preview"
    )
    assert options.headers == {"api-key": "azure-key"}
    assert "model" not in options.params


def test_empty_api_url_without_azure_uses_local_default(monkeypatch):
    monkeypatch.delenv("AZURE_API_BASE", raising=False)
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.setenv("PICTURE_DESCRIPTION_API_URL", "")

    example = _load_example_module()

    options = example._build_picture_description_options()

    assert str(options.url) == "http://localhost:8000/v1/chat/completions"
