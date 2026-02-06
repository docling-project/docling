To use Docling, simply install `docling` from your Python package manager, e.g. pip:

```bash
pip install docling
```

Or using Astral `uv`:

```bash
uv add docling
```

Works on macOS, Linux, and Windows, with support for both x86_64 and arm64 architectures.

??? "Alternative PyTorch distributions"

    The Docling models depend on the [PyTorch](https://pytorch.org/) library.
    Depending on your architecture, you might want to use a different distribution of `torch`.
    For example, you might want support for different accelerator or for a cpu-only version.
    All the different ways for installing `torch` are listed on their website <https://pytorch.org/>.

    One common situation is the installation on Linux systems with cpu-only support.
    In this case, we suggest the installation of Docling with the following options

    ```bash
    # Example for installing on the Linux cpu-only version
    pip install docling --extra-index-url https://download.pytorch.org/whl/cpu
    ```

??? "Installation on macOS Intel (x86_64)"

    When installing Docling on macOS with Intel processors, you might encounter errors with PyTorch compatibility.
    This happens because newer PyTorch versions (2.6.0+) no longer provide wheels for Intel-based Macs.

    If you're using an Intel Mac, install Docling with compatible PyTorch
    **Note:** PyTorch 2.2.2 requires Python 3.12 or lower. Make sure you're not using Python 3.13+.

    ```bash
    # For uv users
    uv add torch==2.2.2 torchvision==0.17.2 docling

    # For pip users
    pip install "docling[mac_intel]"

    # For Poetry users
    poetry add docling
    ```

## CPU-only setup with uv

When using `uv` for dependency management, you can configure CPU-only PyTorch builds through your `pyproject.toml`. This is particularly useful for server deployments or environments without GPU support.

Add the following configuration to your `pyproject.toml`:

```toml
[project]
name = "your-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "docling>=2.71.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    # ... other dependencies
]

[tool.uv.sources]
# Force PyTorch packages to use CPU-only builds
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Then sync your dependencies:

```bash
uv lock
uv sync
```

This configuration ensures that `torch` and `torchvision` are installed from the CPU-only PyTorch index, significantly reducing installation size and avoiding CUDA dependencies.

## Available extras

The `docling` package is designed to offer a working solution for the Docling default options.
Some Docling functionalities require additional third-party packages and are therefore installed only if selected as extras (or installed independently).

The following table summarizes the extras available in the `docling` package. They can be activated with:
`pip install "docling[NAME1,NAME2]"` or `uv add "docling[NAME1,NAME2]"`

| Extra       | Description                                                                                                                                    |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `asr`       | Installs dependencies for running the ASR pipeline.                                                                                            |
| `vlm`       | Installs dependencies for running the VLM pipeline.                                                                                            |
| `easyocr`   | Installs the [EasyOCR](https://github.com/JaidedAI/EasyOCR) OCR engine.                                                                        |
| `tesserocr` | Installs the Tesseract binding for using it as OCR engine.                                                                                     |
| `ocrmac`    | Installs the OcrMac OCR engine.                                                                                                                |
| `rapidocr`  | Installs the [RapidOCR](https://github.com/RapidAI/RapidOCR) OCR engine with [onnxruntime](https://github.com/microsoft/onnxruntime/) backend. |

### OCR engines

Docling supports multiple OCR engines for processing scanned documents. The current version provides
the following engines.

| Engine                                              | Installation                                                                                                                                                                       | Usage                    |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR)      | `easyocr` extra or via `pip install easyocr`.                                                                                                                                      | `EasyOcrOptions`         |
| Tesseract                                           | System dependency. See description for Tesseract and Tesserocr below.                                                                                                              | `TesseractOcrOptions`    |
| Tesseract CLI                                       | System dependency. See description below.                                                                                                                                          | `TesseractCliOcrOptions` |
| OcrMac                                              | System dependency. See description below.                                                                                                                                          | `OcrMacOptions`          |
| [RapidOCR](https://github.com/RapidAI/RapidOCR)     | `rapidocr` extra can or via `pip install rapidocr onnxruntime`                                                                                                                     | `RapidOcrOptions`        |
| [OnnxTR](https://github.com/felixdittrich92/OnnxTR) | Can be installed via the plugin system `pip install "docling-ocr-onnxtr[cpu]"`. Please take a look at [docling-OCR-OnnxTR](https://github.com/felixdittrich92/docling-OCR-OnnxTR). | `OnnxtrOcrOptions`       |

The Docling `DocumentConverter` allows to choose the OCR engine with the `ocr_options` settings. For example

```python
from docling.datamodel.base_models import ConversionStatus, PipelineOptions
from docling.datamodel.pipeline_options import PipelineOptions, EasyOcrOptions, TesseractOcrOptions
from docling.document_converter import DocumentConverter

pipeline_options = PipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = TesseractOcrOptions()  # Use Tesseract

doc_converter = DocumentConverter(
    pipeline_options=pipeline_options,
)
```

??? "Tesseract installation"

    [Tesseract](https://github.com/tesseract-ocr/tesseract) is a popular OCR engine which is available
    on most operating systems. For using this engine with Docling, Tesseract must be installed on your
    system, using the packaging tool of your choice. Below we provide example commands.
    After installing Tesseract you are expected to provide the path to its language files using the
    `TESSDATA_PREFIX` environment variable (note that it must terminate with a slash `/`).

    === "macOS (via [Homebrew](https://brew.sh/))"

        ```console
        brew install tesseract leptonica pkg-config
        TESSDATA_PREFIX=/opt/homebrew/share/tessdata/
        echo "Set TESSDATA_PREFIX=${TESSDATA_PREFIX}"
        ```

    === "Debian-based"

        ```console
        apt-get install tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev pkg-config
        TESSDATA_PREFIX=$(dpkg -L tesseract-ocr-eng | grep tessdata$)
        echo "Set TESSDATA_PREFIX=${TESSDATA_PREFIX}"
        ```

    === "RHEL"

        ```console
        dnf install tesseract tesseract-devel tesseract-langpack-eng tesseract-osd leptonica-devel
        TESSDATA_PREFIX=/usr/share/tesseract/tessdata/
        echo "Set TESSDATA_PREFIX=${TESSDATA_PREFIX}"
        ```

    <h4>Linking to Tesseract</h4>
    The most efficient usage of the Tesseract library is via linking. Docling is using
    the [Tesserocr](https://github.com/sirfz/tesserocr) package for this.

    If you get into installation issues of Tesserocr, we suggest using the following
    installation options:

    **pip:**

    ```console
    pip uninstall tesserocr
    pip install --no-binary :all: tesserocr
    ```

    **uv:**

    ```console
    uv pip uninstall tesserocr
    uv pip install --no-binary :all: tesserocr
    ```

## Development setup

To develop Docling features, bugfixes etc., install as follows from your local clone's root dir:

**uv:**

```bash
uv sync --all-extras
```

**pip:**

```bash
pip install -e ".[all]"
```
