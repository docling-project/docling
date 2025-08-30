# Stages and model runtimes

The current architecture is mixing model runtime (framework, inline/remote, etc) with the stage definition.
The actual choice is done by `kind` fields in the respective `..Options` object.

This is now getting into duplication of runtime logic. For example we have 2x implementation for running vision models from transformers, 2x implementation for running via api inference servers, etc.

Requirements for the proposed changes
1. make more code reusable
2. provide easy presets
3. allow to run custom (model) choices without code changes, e.g. a different model which can do the same task
4. make presets discoverable as plugins, (which can also be third-party contributions)
5. plugins should allow easy choices in clients (CLI, APIs, etc)


TODO:
- [ ] table processing example


## Proposal

### Generic model runtimes

Model runtimes are as generic as possible (but very likely some duplicates might still be there).

1. they operate only on basic objects like PIL images and only expose API for batch predictions
2. the prompt is left out of the model runtime, such that they can be reused
3. model runtime are preferably not bound to a model (but they could if very very specific)
4. model runtime could still have some intenal pre-/post-processing, but it should be limited to model internals, e.g. normalization of images to RGB.

Open questions:
a. should __init__ load the models or we prefer lazy loading?

```py
class BaseModelOptions(BaseModel):
    kind: str

#####
class VisionOpenAILikeApi:
    def __init__(self, options):
        ...

    def predict_batch(self, images: Iterable[PILImage], prompt: str) -> Iterable[...]:
        ...

    @classmethod
    def get_options_type(cls) -> Type[BaseModelOptions]:
        return VisionOpenAILikeApiOptions

#####

class VisionHfTransformersOptions(BaseVlmOptions):
    kind: Literal["vision_hf_transformers"] = "vision_hf_transformers"

    repo_id: str
    trust_remote_code: bool = False
    load_in_8bit: bool = True
    llm_int8_threshold: float = 6.0
    quantized: bool = False

    transformers_model_type: TransformersModelType = TransformersModelType.AUTOMODEL
    transformers_prompt_style: TransformersPromptStyle = TransformersPromptStyle.CHAT

    torch_dtype: Optional[str] = None
    supported_devices: List[AcceleratorDevice] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
    ]

    use_kv_cache: bool = True
    max_new_tokens: int = 4096


class VisionHfTransformers:
    def predict_batch(self, images: Iterable[PILImage], prompt: str) -> Iterable[...]:
        ...

#####

class VisionMlx:
    def predict_batch(self, images: Iterable[PILImage], prompt: str) -> Iterable[...]:
        ...

#####

class VisionVllm:
    def predict_batch(self, images: Iterable[PILImage], prompt: str) -> Iterable[...]:
        ...
```

### Model options and instances

```py
class BaseModelOptions(BaseModel):
    kind: str  # needed for the options to model factory
    name: str  # needed for name (e.g. CLI, etc) to options instance to model factory

class VisionOpenAILikeApiOptions(BaseModelOptions):
    kind: Literal["vision_openailike_api"] = "vision_openailike_api"
    name: str


# Instances

QWEN_VL_OLLAMA = VisionOpenAILikeApiOptions(
    name="qwen_vl_ollama",
    api_url="...",
    model_name="qwen_vl.."
)

SMOLDOCLING_LMSTUDIO = VisionOpenAILikeApiOptions(
    name="smoldocling_lms",
    api_url="...",
    model_name="smoldocling.."
)
SMOLDOCLING_MLX = VisionHfTransformersOptions(
    name="smoldocling_mlx",
    repo_id="ds4sd/smoldocling...",
)
SMOLDOCLING_VLLM = ...

```

### Model factories

Level 1: class names
- From Type[BaseModelOptions] --> Model
- No Enum of kind/names, because these options will have mandatory arguments (api_url, repo_id, etc)

Level 2: instance names
- From the name of the instance 
- Expose enum of all names to be used in CLI, etc

### Stage definition

Stages are responsible for
1. **Pre-process** the input (DoclingDocument, Page batch, etc) to the more generic format
   that can be consumed by the models
2. **Post-process** the output of the models into the format it should be saved back


The stage options are linking together:
1. which stage and its own settings, e.g. the model prompt to use
2. `model_options` used to get the model form the factory
3. `model_interpreter_options` used to interpret the model raw response, which depend on the use-case, so it is independent from the model runtime.
    - we could have each stage (or the ones needing it) define their own factory, but also a shared one should be enough.


```py
## Base classes (options, etc)

class BaseStageOptions(BaseModel):
    kind: str
    model_options: BaseModelOptions
    model_interpreter_options  # in the base clas


## Helper base classes

class BaseDocItemImageEnrichment:
    labels: list[DocItemLabel]  # ..or with a simple filter callable (like now)
    image_scale: float
    expansion_factor: float

    ...


## Actual stages

class PictureDescriptionOptions(BaseStageOptions):
    kind: Literal["picture_description"] = "picture_description"
    model_options: BaseModelOptions = ... # default choice, fully instanciated
    ... # other options

class PictureDescription(BaseDocItemImageEnrichment):
    labels = [PictureItem]
    ...

    def __init__(self, options, ...):
        ...

class CodeUnderstanding(BaseDocItemImageEnrichment):
    labels = [CodeItem]
    ...

    def __init__(self, options, ...):
        ...

class VisionConvertOptions(BaseStageOptions):
    kind: Literal["picture_description"] = "vision_converter"
    model_options: BaseModelOptions = ... # default choice, fully instanciated
    

class VisionConvert:
    """Equivalent to the VlmModel now for DocTags or Markdown"""
    ...

    def __init__(self, options, ...):
        ...
```


### Usage

#### SDK

```py
# Raw inputs
pipeline_options.STAGE_options = PictureDescriptionOptions(
    model_options=VisionOpenAILikeApi(
        api_url="my fancy url",
        model_name="qwen_vl",
    ),
    prompt="Write a few sentences which describe in details this image. If it is a diagram also provide some numeric key highlights."
)

# Using presets
pipeline_options.STAGE_options = PictureDescriptionOptions(
    model_options=model_specs.GRANITE_VISION_LMSTUDIO,
    # there will be a default prompt (but not specific to the model!)
)
```

#### CLI

We could make the options use `--stage-NAME-X` or directly `--NAME-X`.

```sh
# Default options
docling --enrich-picture-description

# Change model (only from preset)
docling --enrich-picture-description \
    --stage-picture-description-model=qwen_vl \
    --stage-picture-description-prompt="..."
```


### Open points

Some minor open questions

1. Should we move the accelerator options in the model_options?
2. Where should the batch_size be?

### Weaknesses

Should we consider storing presets of the full Stage options? Will this quickly become too complex?


## Status

Proposed
