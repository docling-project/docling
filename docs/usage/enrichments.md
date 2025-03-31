Docling allows to enrich the conversion pipeline with additional steps which process specific document components,
e.g. code blocks, pictures, etc. The extra steps usually require extra models executions which may increase
the processing time consistently. For this reason most enrichment models are disabled by default.

The following table provides an overview of the default enrichment models available in Docling.

| Feature | Parameter | Processed item | Description |
| ------- | --------- | ---------------| ----------- |
| Code understanding | `do_code_enrichment` | `CodeItem` | See [docs below](#code-understanding). |
| Formula understanding | `do_formula_enrichment` | `TextItem` with label `FORMULA` | See [docs below](#formula-understanding). |
| Picture classification | `do_picture_classification` | `PictureItem` | See [docs below](#picture-classification). |
| Picture description | `do_picture_description` | `PictureItem` | See [docs below](#picture-description). |


## Enrichments details

### Code understanding

The code understanding step allows to use advance parsing for code blocks found in the document.
This enrichment model also set the `code_language` property of the `CodeItem`.

Model specs: see the [`CodeFormula` model card](https://huggingface.co/ds4sd/CodeFormula).

Example command line:

```sh
docling --enrich-code FILE
```

Example code:

```py
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions()
pipeline_options.do_code_enrichment = True

converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
})

result = converter.convert("https://arxiv.org/pdf/2501.17887")
doc = result.document
```

### Formula understanding

The formula understanding step will analize the equation formulas in documents and extract their LaTeX representation.
The HTML export functions in the DoclingDocument will leverage the formula and visualize the result using the mathml html syntax.

Model specs: see the [`CodeFormula` model card](https://huggingface.co/ds4sd/CodeFormula).

Example command line:

```sh
docling --enrich-formula FILE
```

Example code:

```py
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions()
pipeline_options.do_formula_enrichment = True

converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
})

result = converter.convert("https://arxiv.org/pdf/2501.17887")
doc = result.document
```

### Picture classification

The picture classification step classifies the `PictureItem` elements in the document with the `DocumentFigureClassifier` model.
This model is specialized to understand the classes of pictures found in documents, e.g. different chart types, flow diagrams,
logos, signatures, etc.

Model specs: see the [`DocumentFigureClassifier` model card](https://huggingface.co/ds4sd/DocumentFigureClassifier).

Example command line:

```sh
docling --enrich-picture-classes FILE
```

Example code:

```py
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 2
pipeline_options.do_picture_classification = True

converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
})

result = converter.convert("https://arxiv.org/pdf/2501.17887")
doc = result.document
```


### Picture description

The picture description step allows to annotate a picture with a vision model. This is also known as a "captioning" task.
The Docling pipeline allows to load and run models completely locally as well as connecting to remote API which support the chat template.
Below follow a few examples on how to use some common vision model and remote services.


```py
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions()
pipeline_options.do_picture_description = True

converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
})

result = converter.convert("https://arxiv.org/pdf/2501.17887")
doc = result.document

```

#### Granite Vision model

Model specs: see the [`ibm-granite/granite-vision-3.1-2b-preview` model card](https://huggingface.co/ibm-granite/granite-vision-3.1-2b-preview).

Usage in Docling:

```py
from docling.datamodel.pipeline_options import granite_picture_description

pipeline_options.picture_description_options = granite_picture_description
```

#### SmolVLM model

Model specs: see the [`HuggingFaceTB/SmolVLM-256M-Instruct` model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct).

Usage in Docling:

```py
from docling.datamodel.pipeline_options import smolvlm_picture_description

pipeline_options.picture_description_options = smolvlm_picture_description
```

#### Other vision models

The option class `PictureDescriptionVlmOptions` allows to use any another model from the Hugging Face Hub.

```py
from docling.datamodel.pipeline_options import PictureDescriptionVlmOptions

pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
    repo_id="",  # <-- add here the Hugging Face repo_id of your favorite VLM
    prompt="Describe the image in three sentences. Be consise and accurate.",
)
```

#### Remote vision model

The option class `PictureDescriptionApiOptions` allows to use models hosted on remote platforms, e.g.
on local endpoints served by [VLLM](https://docs.vllm.ai), [Ollama](https://ollama.com/) and others,
or cloud providers like [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai), etc.

_Note: in most cases this option will send your data to the remote service provider._

Usage in Docling:

```py
from docling.datamodel.pipeline_options import PictureDescriptionApiOptions

# Enable connections to remote services
pipeline_options.enable_remote_services=True  # <-- this is required!

# Example using a model running locally, e.g. via VLLM
# $ vllm serve MODEL_NAME
pipeline_options.picture_description_options = PictureDescriptionApiOptions(
    url="http://localhost:8000/v1/chat/completions",
    params=dict(
        model="MODEL NAME",
        seed=42,
        max_completion_tokens=200,
    ),
    prompt="Describe the image in three sentences. Be consise and accurate.",
    timeout=90,
)
```

End-to-end code snippets for cloud providers are available in the examples section:

- [IBM watsonx.ai](../examples/pictures_description_api.py)


## Develop new enrichment models

Beside looking at the implementation of all the models listed above, the Docling documentation has a few examples
dedicated to the implementation of enrichment models.

- [Develop picture enrichment](../examples/develop_picture_enrichment.py)
- [Develop formula enrichment](../examples/develop_formula_understanding.py)
