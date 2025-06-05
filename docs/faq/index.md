# FAQ

This is a collection of FAQ collected from the user questions on <https://github.com/docling-project/docling/discussions>.


??? question "Is Python 3.13 supported?"

    ### Is Python 3.13 supported?

    Python 3.13 is supported from Docling 2.18.0.


??? question "Install conflicts with numpy (python 3.13)"

    ### Install conflicts with numpy (python 3.13)

    When using `docling-ibm-models>=2.0.7` and `deepsearch-glm>=0.26.2` these issues should not show up anymore.
    Docling supports numpy versions `>=1.24.4,<3.0.0` which should match all usages.

    **For older versions**

    This has been observed installing docling and langchain via poetry.

    ```
    ...
    Thus, docling (>=2.7.0,<3.0.0) requires numpy (>=1.26.4,<2.0.0).
    So, because ... depends on both numpy (>=2.0.2,<3.0.0) and docling (^2.7.0), version solving failed.
    ```

    Numpy is only adding Python 3.13 support starting in some 2.x.y version. In order to prepare for 3.13, Docling depends on a 2.x.y for 3.13, otherwise depending an 1.x.y version. If you are allowing 3.13 in your pyproject.toml, Poetry will try to find some way to reconcile Docling's numpy version for 3.13 (some 2.x.y) with LangChain's version for that (some 1.x.y) — leading to the error above.

    Check if Python 3.13 is among the Python versions allowed by your pyproject.toml and if so, remove it and try again.
    E.g., if you have python = "^3.10", use python = ">=3.10,<3.13" instead.

    If you want to retain compatibility with python 3.9-3.13, you can also use a selector in pyproject.toml similar to the following

    ```toml
    numpy = [
        { version = "^2.1.0", markers = 'python_version >= "3.13"' },
        { version = "^1.24.4", markers = 'python_version < "3.13"' },
    ]
    ```

    Source: Issue [#283](https://github.com/docling-project/docling/issues/283#issuecomment-2465035868)


??? question "Is macOS x86_64 supported?"

    ### Is macOS x86_64 supported?

    Yes, Docling (still) supports running the standard pipeline on macOS x86_64.

    However, users might get into a combination of incompatible dependencies on a fresh install.
    Because Docling depends on PyTorch which dropped support for macOS x86_64 after the 2.2.2 release,
    and this old version of PyTorch works only with NumPy 1.x, users **must** ensure the correct NumPy version is running.

    ```shell
    pip install docling "numpy<2.0.0"
    ```

    Source: Issue [#1694](https://github.com/docling-project/docling/issues/1694).


??? question "Are text styles (bold, underline, etc) supported?"

    ### Are text styles (bold, underline, etc) supported?

    Currently text styles are not supported in the `DoclingDocument` format.
    If you are interest in contributing this feature, please open a discussion topic to brainstorm on the design.

    _Note: this is not a simple topic_


??? question "How do I run completely offline?"

    ### How do I run completely offline?

    Docling is not using any remote service, hence it can run in completely isolated air-gapped environments.

    The only requirement is pointing the Docling runtime to the location where the model artifacts have been stored.

    For example

    ```py

    pipeline_options = PdfPipelineOptions(artifacts_path="your location")
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    ```

    Source: Issue [#326](https://github.com/docling-project/docling/issues/326)


??? question " Which model weights are needed to run Docling?"
    ### Which model weights are needed to run Docling?

    Model weights are needed for the AI models used in the PDF pipeline. Other document types (docx, pptx, etc) do not have any such requirement.

    For processing PDF documents, Docling requires the model weights from <https://huggingface.co/ds4sd/docling-models>.

    When OCR is enabled, some engines also require model artifacts. For example EasyOCR, for which Docling has [special pipeline options](https://github.com/docling-project/docling/blob/main/docling/datamodel/pipeline_options.py#L68) to control the runtime behavior.


??? question "SSL error downloading model weights"

    ### SSL error downloading model weights

    ```
    URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)>
    ```

    Similar SSL download errors have been observed by some users. This happens when model weights are fetched from Hugging Face.
    The error could happen when the python environment doesn't have an up-to-date list of trusted certificates.

    Possible solutions were

    - Update to the latest version of [certifi](https://pypi.org/project/certifi/), i.e. `pip install --upgrade certifi`
    - Use [pip-system-certs](https://pypi.org/project/pip-system-certs/) to use the latest trusted certificates on your system.
    - Set environment variables `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` to the value of `python -m certifi`:
        ```
        CERT_PATH=$(python -m certifi)
        export SSL_CERT_FILE=${CERT_PATH}
        export REQUESTS_CA_BUNDLE=${CERT_PATH}
        ```


??? question "Which OCR languages are supported?"

    ### Which OCR languages are supported?

    Docling supports multiple OCR engine, each one has its own list of supported languages.
    Here is a collection of links to the original OCR engine's documentation listing the OCR languages.

    - [EasyOCR](https://www.jaided.ai/easyocr/)
    - [Tesseract](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html)
    - [RapidOCR](https://rapidai.github.io/RapidOCRDocs/blog/2022/09/28/%E6%94%AF%E6%8C%81%E8%AF%86%E5%88%AB%E8%AF%AD%E8%A8%80/)
    - [Mac OCR](https://github.com/straussmaximilian/ocrmac/tree/main?tab=readme-ov-file#example-select-language-preference)

    Setting the OCR language in Docling is done via the OCR pipeline options:

    ```py
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.ocr_options.lang = ["fr", "de", "es", "en"]  # example of languages for EasyOCR
    ```


??? question "Some images are missing from MS Word and Powerpoint"

    ### Some images are missing from MS Word and Powerpoint

    The image processing library used by Docling is able to handle embedded WMF images only on Windows platform.
    If you are on other operating systems, these images will be ignored.


??? question "`HybridChunker` triggers warning: 'Token indices sequence length is longer than the specified maximum sequence length for this model'"

    ### `HybridChunker` triggers warning: 'Token indices sequence length is longer than the specified maximum sequence length for this model'

    **TLDR**:
    In the context of the `HybridChunker`, this is a known & ancitipated "false alarm".

    **Details**:

    Using the [`HybridChunker`](../concepts/chunking.md#hybrid-chunker) often triggers a warning like this:
    > Token indices sequence length is longer than the specified maximum sequence length for this model (531 > 512). Running this sequence through the model will result in indexing errors

    This is a warning that is emitted by transformers, saying that actually *running this sequence through the model* will result in indexing errors, i.e. the problematic case is only if one indeed passes the particular sequence through the (embedding) model.

    In our case though, this occurs as a "false alarm", since what happens is the following:

    - the chunker invokes the tokenizer on a potentially long sequence (e.g. 530 tokens as mentioned in the warning) in order to count its tokens, i.e. to assess if it is short enough. At this point transformers already emits the warning above!
    - whenever the sequence at hand is oversized, the chunker proceeds to split it (but the transformers warning has already been shown nonetheless)

    What is important is the actual token length of the produced chunks.
    The snippet below can be used for getting the actual maximum chunk size (for users wanting to confirm that this does not exceed the model limit):

    ```python
    chunk_max_len = 0
    for i, chunk in enumerate(chunks):
        ser_txt = chunker.serialize(chunk=chunk)
        ser_tokens = len(tokenizer.tokenize(ser_txt))
        if ser_tokens > chunk_max_len:
            chunk_max_len = ser_tokens
        print(f"{i}\t{ser_tokens}\t{repr(ser_txt[:100])}...")
    print(f"Longest chunk yielded: {chunk_max_len} tokens")
    print(f"Model max length: {tokenizer.model_max_length}")
    ```

    Also see [docling#725](https://github.com/docling-project/docling/issues/725).

    Source: Issue [docling-core#119](https://github.com/docling-project/docling-core/issues/119)


??? question "How to use flash attention?"

    ### How to use flash attention?

    When running models in Docling on CUDA devices, you can enable the usage of the Flash Attention2 library.

    Using environment variables:

    ```
    DOCLING_CUDA_USE_FLASH_ATTENTION2=1
    ```

    Using code:

    ```python
    from docling.datamodel.accelerator_options import (
        AcceleratorOptions,
    )

    pipeline_options = VlmPipelineOptions(
        accelerator_options=AcceleratorOptions(cuda_use_flash_attention2=True)
    )
    ```

    This requires having the [flash-attn](https://pypi.org/project/flash-attn/) package installed. Below are two alternative ways for installing it:

    ```shell
    # Building from sources (required the CUDA dev environment)
    pip install flash-attn

    # Using pre-built wheels (not available in all possible setups)
    FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn
    ```
