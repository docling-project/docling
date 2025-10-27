# GPU support

## Achieving Optimal GPU Performance with Docling

This guide describes how to maximize GPU performance for Docling pipelines. It covers device selection, pipeline differences, and provides example snippets for configuring batch size and concurrency in the VLM pipeline for both Linux and Windows.

### Standard Pipeline

Enable GPU acceleration by configuring the accelerator device and concurrency options using Docling's API:

```python
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

# Configure accelerator options for GPU
accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.CUDA,  # or AcceleratorDevice.AUTO
)
```

Batch size and concurrency for document processing are controlled for each stage of the pipeline as:

```python
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)

pipeline_options = ThreadedPdfPipelineOptions(
    ocr_batch_size=32,  # default 4
    layout_batch_size=32,  # default 4
    table_batch_size=4,  # currently not using GPU batching
)
```

Setting a higher `page_batch_size` will run the Docling models (in particular the layout detection stage) with a GPU batch inference mode.

For a complete example see [gpu_standard_pipeline.py](../examples/gpu_standard_pipeline.py).


### VLM Pipeline

For best GPU utilization, use a local inference server. Docling supports inference servers which exposes the OpenAI-compatible chat completion endpoints. For example:

- vllm: `http://localhost:8000/v1/chat/completions` (available only on Linux)
- LM Studio: `http://localhost:1234/v1/chat/completions` (available both on Linux and Windows)
- Ollama: `http://localhost:11434/v1/chat/completions` (available both on Linux and Windows)

Configure the VLM pipeline using Docling's VLM options:

```python
from docling.datamodel.pipeline_options import VlmPipelineOptions

vlm_options = VlmPipelineOptions(
    enable_remote_services=True,
    vlm_options={
        "url": "http://localhost:8000/v1/chat/completions",  # or any other compatible endpoint
        "params": {
            "model": "ibm-granite/granite-docling-258M",
            "max_tokens": 4096,
        },
        "concurrency": 20,  # default is 1
        "prompt": "Convert this page to docling.",
        "timeout": 90,
    }
)
```

Additionally to the concurrency, we also have to set the `page_batch_size` Docling parameter. Make sure to set `settings.perf.page_batch_size >= vlm_options.concurrency`.

```python
from docling.datamodel.settings import settings

settings.perf.page_batch_size = 20  # default is 4
```

For a complete example see [gpu_vlm_pipeline.py](../examples/gpu_vlm_pipeline.py).


#### Available models

Both LM Studio and Ollama rely on llama.cpp as runtime engine. For using this engine, models have to be converted to the gguf format.

Here is a list of known models which are available in gguf format and how to use them.

TBA.

## Performance results

TBA.

