"""Transformers-based VLM inference engine."""

import importlib.metadata
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import torch
from PIL.Image import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedModel,
    ProcessorMixin,
    StoppingCriteriaList,
    StopStringCriteria,
)

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import (
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.runtimes._utils import (
    extract_generation_stoppers,
    preprocess_image_batch,
    resolve_model_artifacts_path,
)
from docling.models.runtimes.base import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
)
from docling.models.utils.generation_utils import (
    GenerationStopper,
    HFStoppingCriteriaWrapper,
)
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin
from docling.utils.accelerator_utils import decide_device

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class TransformersVlmEngine(BaseVlmEngine, HuggingFaceModelDownloadMixin):
    """HuggingFace Transformers engine for VLM inference.

    This engine uses the transformers library to run vision-language models
    locally on CPU, CUDA, or XPU devices.
    """

    def __init__(
        self,
        options: TransformersVlmEngineOptions,
        accelerator_options: Optional[AcceleratorOptions] = None,
        artifacts_path: Optional[Path] = None,
        model_config: Optional[EngineModelConfig] = None,
    ):
        """Initialize the Transformers engine.

        Args:
            options: Transformers-specific runtime options
            accelerator_options: Hardware accelerator configuration
            artifacts_path: Path to cached model artifacts
            model_config: Model configuration (repo_id, revision, extra_config)
        """
        super().__init__(options, model_config=model_config)
        self.options: TransformersVlmEngineOptions = options
        self.accelerator_options = accelerator_options or AcceleratorOptions()
        self.artifacts_path = artifacts_path

        # These will be set during initialization
        self.device: Optional[str] = None
        self.processor: Optional[ProcessorMixin] = None
        self.vlm_model: Optional[PreTrainedModel] = None
        self.generation_config: Optional[GenerationConfig] = None

        # Initialize immediately if model_config is provided
        if self.model_config is not None:
            self.initialize()

    def initialize(self) -> None:
        """Initialize the Transformers model and processor."""
        if self._initialized:
            return

        _log.info("Initializing Transformers VLM inference engine...")

        # Determine device
        supported_devices = [
            AcceleratorDevice.CPU,
            AcceleratorDevice.CUDA,
            AcceleratorDevice.XPU,
        ]
        self.device = decide_device(
            self.options.device or self.accelerator_options.device,
            supported_devices=supported_devices,
        )
        _log.info(f"Using device: {self.device}")

        # Load model if model_config is provided
        if self.model_config is not None and self.model_config.repo_id is not None:
            repo_id = self.model_config.repo_id
            revision = self.model_config.revision or "main"

            # Get model_type from extra_config
            model_type = self.model_config.extra_config.get(
                "transformers_model_type",
                TransformersModelType.AUTOMODEL,
            )

            _log.info(
                f"Loading model {repo_id} (revision: {revision}, "
                f"model_type: {model_type.value})"
            )
            self._load_model_for_repo(repo_id, revision=revision, model_type=model_type)

        self._initialized = True

    def _load_model_for_repo(
        self,
        repo_id: str,
        revision: str = "main",
        model_type: TransformersModelType = TransformersModelType.AUTOMODEL,
    ) -> None:
        """Load model and processor for a specific repository.

        Args:
            repo_id: HuggingFace repository ID
            revision: Model revision
            model_type: Type of model architecture
        """
        # Check for Phi-4 compatibility
        transformers_version = importlib.metadata.version("transformers")
        if (
            repo_id == "microsoft/Phi-4-multimodal-instruct"
            and transformers_version >= "4.52.0"
        ):
            raise NotImplementedError(
                f"Phi 4 only works with transformers<4.52.0 but you have {transformers_version=}. "
                f"Please downgrade by running: pip install -U 'transformers<4.52.0'"
            )

        # Download or locate model artifacts using shared utility
        def download_wrapper(repo_id: str, revision: str) -> Path:
            return self.download_models(repo_id, revision=revision)

        artifacts_path = resolve_model_artifacts_path(
            repo_id=repo_id,
            revision=revision,
            artifacts_path=self.artifacts_path,
            download_fn=download_wrapper,
        )

        # Setup quantization if needed
        quantization_config: Optional[BitsAndBytesConfig] = None
        if self.options.quantized:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.options.load_in_8bit,
                llm_int8_threshold=self.options.llm_int8_threshold,
            )

        # Select model class
        model_cls: type[
            Union[
                AutoModel,
                AutoModelForCausalLM,
                AutoModelForVision2Seq,
                AutoModelForImageTextToText,
            ]
        ] = AutoModel
        if model_type == TransformersModelType.AUTOMODEL_CAUSALLM:
            model_cls = AutoModelForCausalLM  # type: ignore[assignment]
        elif model_type == TransformersModelType.AUTOMODEL_VISION2SEQ:
            model_cls = AutoModelForVision2Seq  # type: ignore[assignment]
        elif model_type == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT:
            model_cls = AutoModelForImageTextToText  # type: ignore[assignment]

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            artifacts_path,
            trust_remote_code=self.options.trust_remote_code,
            revision=revision,
        )
        self.processor.tokenizer.padding_side = "left"  # type: ignore[union-attr]

        # Load model
        self.vlm_model = model_cls.from_pretrained(
            artifacts_path,
            device_map=self.device,
            dtype=self.options.torch_dtype,
            _attn_implementation=(
                "flash_attention_2"
                if self.device.startswith("cuda")  # type: ignore[union-attr]
                and self.accelerator_options.cuda_use_flash_attention2
                else "sdpa"
            ),
            trust_remote_code=self.options.trust_remote_code,
            revision=revision,
            quantization_config=quantization_config,
        )

        # Compile model (Python < 3.14)
        if sys.version_info < (3, 14):
            self.vlm_model = torch.compile(self.vlm_model)  # type: ignore[assignment]
        else:
            self.vlm_model.eval()

        # Load generation config
        self.generation_config = GenerationConfig.from_pretrained(
            artifacts_path, revision=revision
        )

        _log.info(f"Loaded model {repo_id} (revision: {revision})")

    def predict_batch(self, input_batch: List[VlmEngineInput]) -> List[VlmEngineOutput]:
        """Run inference on a batch of inputs efficiently.

        This method processes multiple images in a single forward pass,
        which is much more efficient than processing them sequentially.

        Args:
            input_batch: List of inputs to process

        Returns:
            List of outputs, one per input
        """
        if not self._initialized:
            self.initialize()

        if not input_batch:
            return []

        # Model should already be loaded via initialize()
        if self.vlm_model is None or self.processor is None:
            raise RuntimeError(
                "Model not loaded. Ensure EngineModelConfig was provided during initialization."
            )

        # Get prompt style from first input's extra config
        first_input = input_batch[0]
        prompt_style = first_input.extra_generation_config.get(
            "transformers_prompt_style",
            TransformersPromptStyle.CHAT,
        )

        # Prepare images using shared utility
        images = preprocess_image_batch([inp.image for inp in input_batch])

        # Prepare prompts
        prompts = []
        for input_data in input_batch:
            # Format prompt
            if prompt_style == TransformersPromptStyle.CHAT:
                # Use structured message format with image placeholder (like legacy implementation)
                # This is required for vision models like Granite Vision to properly tokenize
                # both image features and text tokens
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": input_data.prompt},
                        ],
                    }
                ]
                formatted_prompt = self.processor.apply_chat_template(  # type: ignore[union-attr]
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            elif prompt_style == TransformersPromptStyle.RAW:
                formatted_prompt = input_data.prompt
            else:  # NONE
                formatted_prompt = None

            prompts.append(formatted_prompt)

        # Process batch
        if prompt_style == TransformersPromptStyle.NONE:
            inputs = self.processor(  # type: ignore[misc]
                images,
                return_tensors="pt",
                padding=True,
                **first_input.extra_generation_config.get("extra_processor_kwargs", {}),
            )
        else:
            inputs = self.processor(  # type: ignore[misc]
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                **first_input.extra_generation_config.get("extra_processor_kwargs", {}),
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Setup stopping criteria (use first input's config)
        stopping_criteria_list = StoppingCriteriaList()

        if first_input.stop_strings:
            stopping_criteria_list.append(
                StopStringCriteria(
                    stop_strings=first_input.stop_strings,
                    tokenizer=self.processor.tokenizer,  # type: ignore[union-attr,attr-defined]
                )
            )

        # Add custom stopping criteria using shared utility
        custom_stoppers = extract_generation_stoppers(
            first_input.extra_generation_config
        )
        for stopper in custom_stoppers:
            wrapped_criteria = HFStoppingCriteriaWrapper(
                self.processor.tokenizer,  # type: ignore[union-attr,attr-defined]
                stopper,
            )
            stopping_criteria_list.append(wrapped_criteria)

        # Also handle any HF StoppingCriteria directly passed
        custom_criteria = first_input.extra_generation_config.get(
            "custom_stopping_criteria", []
        )
        for criteria in custom_criteria:
            # Skip GenerationStopper instances (already handled above)
            if not isinstance(criteria, GenerationStopper) and not (
                isinstance(criteria, type) and issubclass(criteria, GenerationStopper)
            ):
                stopping_criteria_list.append(criteria)

        # Filter decoder-specific keys
        decoder_keys = {
            "skip_special_tokens",
            "clean_up_tokenization_spaces",
            "spaces_between_special_tokens",
        }
        generation_config = {
            k: v
            for k, v in first_input.extra_generation_config.items()
            if k not in decoder_keys
            and k
            not in {
                "transformers_model_type",
                "transformers_prompt_style",
                "extra_processor_kwargs",
                "custom_stopping_criteria",
                "revision",
            }
        }
        decoder_config = {
            k: v
            for k, v in first_input.extra_generation_config.items()
            if k in decoder_keys
        }

        # Generate
        gen_kwargs = {
            **inputs,
            "max_new_tokens": first_input.max_new_tokens,
            "use_cache": self.options.use_kv_cache,
            "generation_config": self.generation_config,
            **generation_config,
        }

        if first_input.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = first_input.temperature
        else:
            gen_kwargs["do_sample"] = False

        if stopping_criteria_list:
            gen_kwargs["stopping_criteria"] = stopping_criteria_list

        start_time = time.time()
        with torch.inference_mode():
            generated_ids = self.vlm_model.generate(**gen_kwargs)  # type: ignore[union-attr,operator]
        generation_time = time.time() - start_time

        # Decode
        input_len = inputs["input_ids"].shape[1]
        trimmed_sequences = generated_ids[:, input_len:]

        decode_fn = getattr(self.processor, "batch_decode", None)
        if decode_fn is None and hasattr(self.processor, "tokenizer"):
            decode_fn = self.processor.tokenizer.batch_decode  # type: ignore[union-attr]
        if decode_fn is None:
            raise RuntimeError(
                "Neither processor.batch_decode nor tokenizer.batch_decode is available."
            )

        decoded_texts = decode_fn(trimmed_sequences, **decoder_config)

        # Remove padding
        pad_token = self.processor.tokenizer.pad_token  # type: ignore[union-attr,attr-defined]
        if pad_token:
            decoded_texts = [text.rstrip(pad_token) for text in decoded_texts]

        # Create outputs
        outputs = []
        for i, text in enumerate(decoded_texts):
            outputs.append(
                VlmEngineOutput(
                    text=text,
                    stop_reason="unspecified",
                    metadata={
                        "generation_time": generation_time / len(input_batch),
                        "num_tokens": int(generated_ids[i].shape[0])
                        if i < generated_ids.shape[0]
                        else None,
                        "batch_size": len(input_batch),
                    },
                )
            )

        _log.info(
            f"Batch processed {len(input_batch)} images in {generation_time:.2f}s "
            f"({generation_time / len(input_batch):.2f}s per image)"
        )

        return outputs

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.vlm_model is not None:
            del self.vlm_model
            self.vlm_model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        # Clear CUDA cache if using GPU
        if self.device and self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        _log.info("Transformers runtime cleaned up")
