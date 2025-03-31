To use Docling, simply install `docling` from your Python package manager, e.g. pip:
```bash
pip install docling
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

??? "Alternative OCR engines"

    Docling supports multiple OCR engines for processing scanned documents. The current version provides
    the following engines.

    | Engine | Installation | Usage |
    | ------ | ------------ | ----- |
    | [EasyOCR](https://github.com/JaidedAI/EasyOCR) | Default in Docling or via `pip install easyocr`. | `EasyOcrOptions` |
    | Tesseract | System dependency. See description for Tesseract and Tesserocr below.  | `TesseractOcrOptions` |
    | Tesseract CLI | System dependency. See description below. | `TesseractCliOcrOptions` |
    | OcrMac | System dependency. See description below. | `OcrMacOptions` |
    | [RapidOCR](https://github.com/RapidAI/RapidOCR) | Extra feature not included in Default Docling installation can be installed via `pip install rapidocr_onnxruntime` | `RapidOcrOptions` |

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

    <h3>Tesseract installation</h3>

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
        dnf install tesseract tesseract-devel tesseract-langpack-eng leptonica-devel
        TESSDATA_PREFIX=/usr/share/tesseract/tessdata/
        echo "Set TESSDATA_PREFIX=${TESSDATA_PREFIX}"
        ```

    <h3>Linking to Tesseract</h3>
    The most efficient usage of the Tesseract library is via linking. Docling is using
    the [Tesserocr](https://github.com/sirfz/tesserocr) package for this.

    If you get into installation issues of Tesserocr, we suggest using the following
    installation options:

    ```console
    pip uninstall tesserocr
    pip install --no-binary :all: tesserocr
    ```

    <h3>ocrmac installation</h3>

    [ocrmac](https://github.com/straussmaximilian/ocrmac) is using
    Apple's vision(or livetext) framework as OCR backend.
    For using this engine with Docling, ocrmac must be installed on your system.
    This only works on macOS systems with newer macOS versions (10.15+).

    ```console
    pip install ocrmac
    ```

## Development setup

To develop Docling features, bugfixes etc., install as follows from your local clone's root dir:

```bash
poetry install --all-extras
```
