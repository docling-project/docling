# GPU support

## Achieving Optimal GPU Performance with Docling

This guide describes how to maximize GPU performance for Docling pipelines. It covers device selection, pipeline differences, and provides example snippets for configuring batch size and concurrency in the VLM pipeline for both Linux and Windows.

!!! note

    Improvements and optimizations strategies for maximizing the GPU performance is an
    active topic. Regularly check these guidelines for updates.


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
    ocr_batch_size=64,  # default 4
    layout_batch_size=64,  # default 4
    table_batch_size=4,  # currently not using GPU batching
)
```

Setting a higher `page_batch_size` will run the Docling models (in particular the layout detection stage) with a GPU batch inference mode.

#### Complete example

For a complete example see [gpu_standard_pipeline.py](../examples/gpu_standard_pipeline.py).

#### OCR engines

The current Docling OCR engines rely on third-party libraries, hence GPU support depends on the availability in the respective engines.

The only setup which is known to work at the moment is RapidOCR with the torch backend, which can be enabled via

```py
pipeline_options = PdfPipelineOptions()
pipeline_options.ocr_options = RapidOcrOptions(
    backend="torch",
)
```

More details in the GitHub discussion [#2451](https://github.com/docling-project/docling/discussions/2451).


### VLM Pipeline

For best GPU utilization, use a local inference server. Docling supports inference servers which exposes the OpenAI-compatible chat completion endpoints. For example:

- vllm: `http://localhost:8000/v1/chat/completions` (available only on Linux)
- LM Studio: `http://localhost:1234/v1/chat/completions` (available both on Linux and Windows)
- Ollama: `http://localhost:11434/v1/chat/completions` (available both on Linux and Windows)


#### Start the inference server

Here is an example on how to start the [vllm](https://docs.vllm.ai/) inference server with optimum parameters for Granite Docling.

```sh
vllm serve ibm-granite/granite-docling-258M \
  --host 127.0.0.1 --port 8000 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.9
```

#### Configure Docling

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
        "concurrency": 64,  # default is 1
        "prompt": "Convert this page to docling.",
        "timeout": 90,
    }
)
```

Additionally to the concurrency, we also have to set the `page_batch_size` Docling parameter. Make sure to set `settings.perf.page_batch_size >= vlm_options.concurrency`.

```python
from docling.datamodel.settings import settings

settings.perf.page_batch_size = 64  # default is 4
```

#### Complete example

For a complete example see [gpu_vlm_pipeline.py](../examples/gpu_vlm_pipeline.py).


#### Available models

Both LM Studio and Ollama rely on llama.cpp as runtime engine. For using this engine, models have to be converted to the gguf format.

Here is a list of known models which are available in gguf format and how to use them.

TBA.

## Performance results

### Test data

| | PDF doc | [ViDoRe V3 HR](https://huggingface.co/datasets/vidore/vidore_v3_hr) |
| - | - | - |
| Num docs | 1 | 14 |
| Num pages | 192 | 1110 |
| Num tables | 95 | 258 |
| Format type | PDF | Parquet of images |


### Test infrastructure

| | g6e.2xlarge | RTX 5090 | RTX 5070 |
| - | - | - | - |
| Description | AWS instance `g6e.2xlarge` | Linux bare metal machine | Windows 11 bare metal machine |
| CPU | 8 vCPUs, AMD EPYC 7R13 | 16 vCPU, AMD Ryzen 7 9800 | 16 vCPU, AMD Ryzen 7 9800 |
| RAM | 64GB | 128GB | 64GB |
| GPU | NVIDIA L40S 48GB | NVIDIA GeForce RTX 5090 | NVIDIA GeForce RTX 5070 |
| CUDA Version | 13.0, driver 580.95.05 | 13.0, driver 580.105.08 | 13.0, driver 581.57 |


### Results

<table>
  <thead>
    <tr><th rowspan="2">Pipeline</th><th colspan="2">g6e.2xlarge</th><th colspan="2">RTX 5090</th><th colspan="2">RTX 5070</th></tr>
    <tr><th>PDF doc</th><th>ViDoRe V3 HR</th><th>PDF doc</th><th>ViDoRe V3 HR</th><th>PDF doc</th><th>ViDoRe V3 HR</th></tr>
  </thead>
  <tbody>
    <tr><td>Standard - Inline (no OCR)</td><td>3.1 pages/second</td><td>-</td><td>7.9 pages/second<br /><small><em>[cpu-only]* 1.5 pages/second</em></small></td><td>-</td><td>4.2 pages/second<br /><small><em>[cpu-only]* 1.2 pages/second</em></small></td><td>-</td></tr>
    <tr><td>Standard - Inline (with OCR)</td><td></td><td></td><td>tba</td><td>1.6 pages/second</td><td>tba</td><td>1.1 pages/second</td></tr>
    <tr><td>VLM - Inference server (GraniteDocling)</td><td>2.4 pages/second</td><td>-</td><td>3.8 pages/second</td><td>3.6-4.5 pages/second</td><td>2.0 pages/second</td><td>2.8-3.2 pages/second</td></tr>
  </tbody>
</table>

_* cpu-only timing computed with 16 pytorch threads._
