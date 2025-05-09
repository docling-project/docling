import os


def _setup_env():
    os.environ["DOCLING_PERF_PAGE_BATCH_SIZE"] = "12"
    os.environ["DOCLING_DEBUG_VISUALIZE_RAW_LAYOUT"] = "True"
    os.environ["DOCLING_ARTIFACTS_PATH"] = "/path/to/artifacts"


def test_settings():
    _setup_env()

    from docling.datamodel.settings import settings

    # Check top level setting
    assert str(settings.artifacts_path) == "/path/to/artifacts"

    # Check nested set via environment variables
    assert settings.perf.page_batch_size == 12
    assert settings.debug.visualize_raw_layout is True

    # Check nested defaults
    assert settings.perf.doc_batch_size == 2
    assert settings.debug.visualize_ocr is False
