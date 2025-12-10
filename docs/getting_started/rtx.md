# RTX GPU Acceleration

TBA. Docling+RTX artwork

Whether you're an AI enthusiast, researcher, or developer working with document processing, this guide will help you unlock the full potential of your NVIDIA RTX GPU with Docling.

By leveraging GPU acceleration, you can achieve up to **6x speedup** compared to CPU-only processing. This dramatic performance improvement makes GPU acceleration especially valuable for processing large batches of documents, handling high-throughput document conversion workflows, or experimenting with advanced document understanding models.

TBA. Performance improvement figure.

## Prerequisites

Before setting up GPU acceleration, ensure you have:

- An NVIDIA RTX GPU (RTX 40/50 series or newer)
- Windows 10/11 or Linux operating system

## Installation Steps

### 1. Install NVIDIA GPU Drivers

First, ensure you have the latest NVIDIA GPU drivers installed:

- **Windows**: Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- **Linux**: Use your distribution's package manager or download from NVIDIA

Verify the installation:

```bash
nvidia-smi
```

This command should display your GPU information and driver version.

### 2. Install CUDA Toolkit

CUDA is NVIDIA's parallel computing platform required for GPU acceleration.

Follow the official installation guide for your operating system at [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads). The installer will guide you through the process and automatically set up the required environment variables.

### 3. Install cuDNN

cuDNN provides optimized implementations for deep learning operations.

Follow the official installation guide at [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn). The guide provides detailed instructions for all supported platforms.

### 4. Install PyTorch with CUDA Support

To use GPU acceleration with Docling, you need to install PyTorch with CUDA support using the special `extra-index-url`:

```bash
# For CUDA 12.8 (current default for PyTorch)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# For CUDA 13.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

!!! note
    The `--index-url` parameter is crucial as it ensures you get the CUDA-enabled version of PyTorch instead of the CPU-only version.

For other CUDA versions and installation options, refer to the [PyTorch Installation Matrix](https://pytorch.org/get-started/locally/).

Verify PyTorch CUDA installation:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 5. Install and Run Docling

Install Docling with all dependencies:

```bash
pip install docling
```

**That's it!** Docling will automatically detect and use your RTX GPU when available. No additional configuration is required for basic usage.

```python
from docling.document_converter import DocumentConverter

# Docling automatically uses GPU when available
converter = DocumentConverter()
result = converter.convert("document.pdf")
```

<details>
<summary><b>Advanced: Tuning GPU Performance</b></summary>

For optimal GPU performance with large document batches, you can adjust batch sizes and explicitly configure the accelerator:

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions

# Explicitly configure GPU acceleration
accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.CUDA,  # Use CUDA for NVIDIA GPUs
)

# Configure pipeline for optimal GPU performance
pipeline_options = ThreadedPdfPipelineOptions(
    ocr_batch_size=64,      # Increase batch size for GPU
    layout_batch_size=64,   # Increase batch size for GPU
    table_batch_size=4,
)

# Create converter with custom settings
converter = DocumentConverter(
    accelerator_options=accelerator_options,
    pipeline_options=pipeline_options,
)

# Convert documents
result = converter.convert("document.pdf")
```

Adjust batch sizes based on your GPU memory (see Performance Optimization Tips below).

</details>

## GPU-Accelerated VLM Pipeline

For maximum performance with Vision Language Models (VLM), you can run a local inference server on your RTX GPU. This approach provides significantly better throughput than inline VLM processing.

### Linux: Using vLLM (Recommended)

vLLM provides the best performance for GPU-accelerated VLM inference. Start the vLLM server with optimized parameters:

```bash
vllm serve ibm-granite/granite-docling-258M \
  --host 127.0.0.1 --port 8000 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.9
```

### Windows: Using llama-server

On Windows, you can use `llama-server` from llama.cpp for GPU-accelerated VLM inference:

#### Installation

1. Download the latest llama.cpp release from the [GitHub releases page](https://github.com/ggml-org/llama.cpp/releases)
2. Extract the archive and locate `llama-server.exe`

#### Launch Command

```powershell
llama-server.exe `
  --hf-repo ibm-granite/granite-docling-258M-GGUF `
  -cb `
  -ngl -1 `
  --port 8000 `
  --context-shift `
  -np 16 -c 131072
```

!!! note "Performance Comparison"
    vLLM delivers approximately **4x better performance** compared to llama-server. For Windows users seeking maximum performance, consider running vLLM via WSL2 (Windows Subsystem for Linux). See [vLLM on RTX 5090 guide](https://github.com/BoltzmannEntropy/vLLM-5090) for detailed WSL2 setup instructions.

### Configure Docling for VLM Server

Once your inference server is running, configure Docling to use it:

TODO: simplify the setup with a pre-configured config.

```python
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.settings import settings

# Configure VLM options
vlm_options = VlmPipelineOptions(
    enable_remote_services=True,
    vlm_options={
        "url": "http://localhost:8000/v1/chat/completions",
        "params": {
            "model": "ibm-granite/granite-docling-258M",
            "max_tokens": 4096,
        },
        "concurrency": 64,  # Adjust based on GPU memory
        "prompt": "Convert this page to docling.",
        "timeout": 90,
    }
)

# Set page batch size to match or exceed concurrency
settings.perf.page_batch_size = 64

# Create converter with VLM pipeline
converter = DocumentConverter(
    pipeline_options=vlm_options,
)
```

For more details on VLM pipeline configuration, see the [GPU Support Guide](../usage/gpu.md).

## Performance Optimization Tips

### Batch Size Tuning

Adjust batch sizes based on your GPU memory:

- **RTX 5090 (32GB)**: Use batch sizes of 64-128
- **RTX 4090 (24GB)**: Use batch sizes of 32-64
- **RTX 5070 (12GB)**: Use batch sizes of 16-32

### Memory Management

Monitor GPU memory usage:

```python
import torch

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

## Troubleshooting

### CUDA Out of Memory

If you encounter out-of-memory errors:

1. Reduce batch sizes in `pipeline_options`
2. Process fewer documents concurrently
3. Clear GPU cache between batches:

```python
import torch
torch.cuda.empty_cache()
```

### CUDA Not Available

If `torch.cuda.is_available()` returns `False`:

1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version
4. Ensure your GPU is CUDA-compatible

### Performance Not Improving

If GPU acceleration doesn't improve performance:

1. Increase batch sizes (if memory allows)
2. Ensure you're processing enough documents to benefit from GPU parallelization
3. Check GPU utilization: `nvidia-smi -l 1`
4. Verify PyTorch is using GPU: `torch.cuda.is_available()`

## Additional Resources

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch CUDA Installation Guide](https://pytorch.org/get-started/locally/)
- [Docling GPU Support Guide](../usage/gpu.md)
- [GPU Performance Examples](../examples/gpu_standard_pipeline.py)
