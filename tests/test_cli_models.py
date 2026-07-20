"""
Tests for the `docling-tools models download` and `docling-tools models download-hf-repo` commands.
These tests use a mock HTTP server to intercept HuggingFace Hub requests and verify
that tokens are passed correctly.
"""

from unittest import mock
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from typing import List, Tuple
import os
import pytest

from docling.cli.models import app
from docling.exceptions import DoclingModelDownloadError, DoclingMultiModelDownloadError

runner = CliRunner()

# ==========================================
# TEST CASES
# ==========================================


@pytest.fixture
def mock_snapshot_download():
    """
    Patches `snapshot_download` where it is imported.
    Returns a Mock object that captures all calls and prevents actual downloads.
    """
    target_path = "docling.models.utils.hf_model_download.snapshot_download"
    with patch(target_path) as mock_download:
        mock_download.return_value = "/dummy/path"
        yield mock_download


@pytest.fixture
def mock_failed_download_hf_model():
    """
    Mocks `download_hf_model` to raise a DoclingModelDownloadError exception to
    simulate a download failure (e.g. invalid token, network error, etc)
    """
    target_path = "docling.cli.models.download_hf_model"
    with patch(target_path) as mock_func:
        mock_func.side_effect = DoclingModelDownloadError(
            "Authentication failed: The provided HuggingFace token is invalid or expired"
        )
        yield mock_func


@pytest.fixture
def mock_failed_download_models():
    """
    Mocks `download_models` to raise a DoclingMulitModelDownloadError exception to
    simulate multi-model download failure (e.g. invalid token, network error, etc)
    """
    failed_models: List[tuple[str, Exception]] = [
        (
            "LayoutModel",
            DoclingModelDownloadError(
                "Authentication failed: The provided Hugging Face token is invalid or unauthorized ..."
            ),
        )
    ]
    target_path = "docling.cli.models.download_models"
    with patch(target_path) as mock_func:
        mock_func.side_effect = DoclingMultiModelDownloadError(failed_models)
        yield mock_func


def test_models_download_help():
    """Test that the models download command help works."""
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    assert "--hf-token" in result.output


def test_models_download_hf_repo_help():
    """Test that the models download-hf-repo command help works."""
    result = runner.invoke(app, ["download-hf-repo", "--help"])
    assert result.exit_code == 0
    assert "--hf-token" in result.output


def test_models_download_with_token_string(mock_snapshot_download):
    """Test models download command with string HF token using mock server."""

    result = runner.invoke(
        app,
        ["download", "--all", "--hf-token", "test_token"],
    )
    # Check that the server received requests
    mock_snapshot_download.assert_called()
    _, kwargs = mock_snapshot_download.call_args

    assert kwargs.get("token") == "test_token"


def test_models_download_with_token_from_env(mock_snapshot_download, monkeypatch):
    """Test models download command with the HF_TOKEN env var."""

    # Set the environment variable for the token
    monkeypatch.setenv("HF_TOKEN", "env_test_token")

    result = runner.invoke(app, ["download", "--all"])

    assert result.exit_code == 0
    mock_snapshot_download.assert_called()
    _, kwargs = mock_snapshot_download.call_args
    assert kwargs.get("token") == "env_test_token"


def test_models_download_without_token_no_auth(mock_snapshot_download, monkeypatch):
    """Test models download command without token."""

    # Ensure no environment token is leaking into this test
    monkeypatch.delenv("HF_TOKEN", raising=False)

    # Run download command without any token provided
    result = runner.invoke(app, ["download", "--all"])

    assert result.exit_code == 0
    mock_snapshot_download.assert_called()

    _, kwargs = mock_snapshot_download.call_args

    assert kwargs.get("token") is None


def test_models_download_with_invalid_token(mock_failed_download_models):
    """
    Test models download command with invalid token returns
    non-zero exit code with an appropriate error message
    """
    invalid_token = "this-is-an-invalid-token"

    result = runner.invoke(app, ["download", "layout", "--hf-token", invalid_token])

    mock_failed_download_models.assert_called()

    # Check if the invalid token propagated
    assert mock_failed_download_models.call_args.kwargs.get("hf_token") == invalid_token

    # Non-zero exit code
    assert result.exit_code == 1, f"Expected exit code 1, but got {result.exit_code}"

    assert "Model download(s) finished with errors" in result.output
    assert "LayoutModel" in result.output  # The key of the model that failed
    assert "Authentication failed" in result.output

    # Verify the conditional helper tip about the token is printed to guide the user
    assert "Tip: One or more downloads failed due to authentication" in result.output
    assert "--hf-token" in result.output


def test_models_download_with_invalid_token_and_quiet_option(
    mock_failed_download_models,
):
    """
    Test models download command with invalid token and quiet flag returns
    non-zero exit code silently
    """
    invalid_token = "this-is-an-invalid-token"

    result = runner.invoke(
        app, ["download", "layout", "--hf-token", invalid_token, "-q"]
    )

    mock_failed_download_models.assert_called()

    # Check if the invalid token propagated
    assert mock_failed_download_models.call_args.kwargs.get("hf_token") == invalid_token

    # Non-zero exit code
    assert result.exit_code == 1, f"Expected exit code 1, but got {result.exit_code}"

    assert "Model download(s) finished with errors" not in result.output
    assert "Authentication failed" not in result.output

    # Verify the conditional helper tip about the token is printed to guide the user
    assert (
        "Tip: One or more downloads failed due to authentication" not in result.output
    )
    assert "--hf-token" not in result.output


def test_models_download_hf_repo_with_token_string(mock_snapshot_download):
    """Test models download command with string HF token using mock server."""

    result = runner.invoke(
        app,
        ["download-hf-repo", "dummy-repo/file1", "--hf-token", "test_token"],
    )
    # Check that the server received requests
    mock_snapshot_download.assert_called()
    _, kwargs = mock_snapshot_download.call_args

    assert kwargs.get("token") == "test_token"
    assert kwargs.get("repo_id") == "dummy-repo/file1"


def test_models_download_hf_repo_with_token_from_env(
    mock_snapshot_download, monkeypatch
):
    """Test models download command with the HF_TOKEN env var."""

    # Set the environment variable for the token
    monkeypatch.setenv("HF_TOKEN", "env_test_token")

    result = runner.invoke(app, ["download-hf-repo", "dummy-repo/file1"])

    assert result.exit_code == 0
    mock_snapshot_download.assert_called()
    _, kwargs = mock_snapshot_download.call_args
    assert kwargs.get("token") == "env_test_token"
    assert kwargs.get("repo_id") == "dummy-repo/file1"


def test_models_download_hf_repo_without_token_no_auth(
    mock_snapshot_download, monkeypatch
):
    """Test models download command without token sends no Authorization header."""

    # Ensure no environment token is leaking into this test
    monkeypatch.delenv("HF_TOKEN", raising=False)

    # Run download command without any token provided
    result = runner.invoke(app, ["download-hf-repo", "dummy-repo/file1"])

    assert result.exit_code == 0
    mock_snapshot_download.assert_called()

    _, kwargs = mock_snapshot_download.call_args

    assert kwargs.get("token") is None
    assert kwargs.get("repo_id") == "dummy-repo/file1"


def test_models_download_hf_repo_with_invalid_token(mock_failed_download_hf_model):
    """
    Test models download-hf-repo command with invalid token returns
    non-zero exit code with an appropriate error message
    """
    invalid_token = "this-is-an-invalid-token"
    repo_id = "docling-project/docling-models"

    result = runner.invoke(
        app, ["download-hf-repo", repo_id, "--hf-token", invalid_token]
    )
    mock_failed_download_hf_model.assert_called()

    # Check if the invalid token propagated
    assert mock_failed_download_hf_model.call_args.kwargs.get("token") == invalid_token

    # Non-zero exit code
    assert result.exit_code == 1, f"Expected exit code 1, but got {result.exit_code}"

    assert "Model download(s) finished with errors" in result.output
    assert repo_id in result.output  # The key of the model that failed
    assert "Authentication failed" in result.output

    # Verify the conditional helper tip about the token is printed to guide the user
    assert "Tip: One or more downloads failed due to authentication" in result.output
    assert "--hf-token" in result.output


def test_models_download_hf_repo_with_invalid_token_and_quiet_option(
    mock_failed_download_hf_model,
):
    """
    Test models download-hf-repo command with invalid token and quiet flag returns
    non-zero exit code silently
    """
    invalid_token = "this-is-an-invalid-token"
    repo_id = "docling-project/docling-models"

    result = runner.invoke(
        app, ["download-hf-repo", repo_id, "--hf-token", invalid_token, "-q"]
    )
    mock_failed_download_hf_model.assert_called()

    # Check if the invalid token propagated
    assert mock_failed_download_hf_model.call_args.kwargs.get("token") == invalid_token

    # Non-zero exit code
    assert result.exit_code == 1, f"Expected exit code 1, but got {result.exit_code}"

    assert "Model download(s) finished with errors" not in result.output
    assert "Authentication failed" not in result.output

    # Verify the conditional helper tip about the token is printed to guide the user
    assert (
        "Tip: One or more downloads failed due to authentication" not in result.output
    )
    assert "--hf-token" not in result.output
