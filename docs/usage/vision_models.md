
The `VlmPipeline` in Docling allows to convert documents end-to-end using a vision-language model.

Docling supports vision-language models which output:

- DocTags (e.g. [SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview)), the preferred choice
- Markdown
- HTML


For running Docling using local models with the `VlmPipeline`:

=== "CLI"

    ```bash
    docling --pipeline vlm FILE
    ```

=== "Python"

    See also the example [minimal_vlm_pipeline.py](./../examples/minimal_vlm_pipeline.py).

    ```python
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
            ),
        }
    )

    doc = converter.convert(source="FILE").document
    ```

## Available local models

By default, the vision-language models are running locally.
Docling allows to choose between the Hugging Face [Transformers](https://github.com/huggingface/transformers) framweork and the [MLX](https://github.com/Blaizzy/mlx-vlm) (for Apple devices with MPS acceleration) one.

The following table reports the models currently available out-of-the-box.

| Model instance | Model | Framework | Device | Num pages | Inference time (sec) |
| ---------------|------ | --------- | ------ | --------- | ---------------------|
| `vlm_model_specs.SMOLDOCLING_TRANSFORMERS` | [ds4sd/SmolDocling-256M-preview](https://huggingface.co/ds4sd/SmolDocling-256M-preview) | `Transformers/AutoModelForVision2Seq` | MPS | 1 |  102.212 |
| `vlm_model_specs.SMOLDOCLING_MLX` | [ds4sd/SmolDocling-256M-preview-mlx-bf16](https://huggingface.co/ds4sd/SmolDocling-256M-preview-mlx-bf16) | `MLX`| MPS | 1 |    6.15453 |
| `vlm_model_specs.QWEN25_VL_3B_MLX` | [mlx-community/Qwen2.5-VL-3B-Instruct-bf16](https://huggingface.co/mlx-community/Qwen2.5-VL-3B-Instruct-bf16)  |  `MLX`| MPS | 1 |   23.4951 |
| `vlm_model_specs.PIXTRAL_12B_MLX` | [mlx-community/pixtral-12b-bf16](https://huggingface.co/mlx-community/pixtral-12b-bf16) |  `MLX` | MPS | 1 |  308.856 |
| `vlm_model_specs.GEMMA3_12B_MLX` | [mlx-community/gemma-3-12b-it-bf16](https://huggingface.co/mlx-community/gemma-3-12b-it-bf16) |  `MLX` | MPS | 1 |  378.486 |
| `vlm_model_specs.GRANITE_VISION_TRANSFORMERS` | [ibm-granite/granite-vision-3.2-2b](https://huggingface.co/ibm-granite/granite-vision-3.2-2b) | `Transformers/AutoModelForVision2Seq` | MPS | 1 |  104.75 |
| `vlm_model_specs.PHI4_TRANSFORMERS` | [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | `Transformers/AutoModelForCasualLM` | CPU | 1 | 1175.67 |
| `vlm_model_specs.PIXTRAL_12B_TRANSFORMERS` | [mistral-community/pixtral-12b](https://huggingface.co/mistral-community/pixtral-12b) | `Transformers/AutoModelForVision2Seq` | CPU | 1 | 1828.21 |

_Inference time is computed on a Macbook M3 Max using the example page `tests/data/pdf/2305.03393v1-pg9.pdf`. The comparision is done with the example [compare_vlm_models.py](./../examples/compare_vlm_models.py)._

For choosing the model, the code snippet above can be extended as follow

```python
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel import vlm_model_specs

pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.SMOLDOCLING_MLX,  # <-- change the model here
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

doc = converter.convert(source="FILE").document
```

### Other models

Other models can be configured by directly providing the Hugging Face `repo_id`, the prompt and a few more options.

For example:

```python
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions, InferenceFramework, TransformersModelType

pipeline_options = VlmPipelineOptions(
    vlm_options=InlineVlmOptions(
        repo_id="ibm-granite/granite-vision-3.2-2b",
        prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
        response_format=ResponseFormat.MARKDOWN,
        inference_framework=InferenceFramework.TRANSFORMERS,
        transformers_model_type=TransformersModelType.AUTOMODEL_VISION2SEQ,
        supported_devices=[
            AcceleratorDevice.CPU,
            AcceleratorDevice.CUDA,
            AcceleratorDevice.MPS,
        ],
        scale=2.0,
        temperature=0.0,
    )
)
```


## Remote models

Additionally to local models, the `VlmPipeline` allows to offload the inference to a remote service hosting the models.
Many remote inference services are provided, the key requirement is to offer an OpenAI-compatible API. This includes vLLM, Ollama, etc.

More examples on how to connect with the remote inference services can be found in the following examples:

- [vlm_pipeline_api_model.py](./../examples/vlm_pipeline_api_model.py)
