"""vLLM-based VLM inference engine for high-throughput serving."""

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import TransformersPromptStyle
from docling.datamodel.vlm_engine_options import VllmVlmEngineOptions
from docling.models.inference_engines.vlm._utils import (
    format_prompt_for_vlm,
    preprocess_image_batch,
    resolve_model_artifacts_path,
)
from docling.models.inference_engines.vlm.base import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
)
from docling.utils.accelerator_utils import decide_device

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class VllmVlmEngine(BaseVlmEngine):
    """vLLM engine for high-throughput VLM inference.

    This engine uses the vLLM library for efficient batched inference
    on CUDA and XPU devices.
    """

    # Allowlist of vLLM SamplingParams arguments (runtime generation controls)
    _VLLM_SAMPLING_KEYS = {
        # Core
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        # Penalties
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        # Stops / outputs
        "stop",
        "stop_token_ids",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        # Search / length
        "n",
        "best_of",
        "length_penalty",
        "early_stopping",
        # Misc
        "logprobs",
        "prompt_logprobs",
        "min_p",
        "seed",
    }

    # Allowlist of vLLM LLM/EngineArgs arguments (engine/load-time controls)
    _VLLM_ENGINE_KEYS = {
        # Model/tokenizer/impl
        "tokenizer",
        "tokenizer_mode",
        "download_dir",
        # Parallelism / memory / lengths
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_batched_tokens",
        "kv_cache_dtype",
        "dtype",
        # Quantization
        "quantization",
        # Multimodal limits
        "limit_mm_per_prompt",
        # Execution toggles
        "enforce_eager",
    }

    def __init__(
        self,
        options: VllmVlmEngineOptions,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Union[Path, str]],
        model_config: Optional["EngineModelConfig"] = None,
    ):
        """Initialize the vLLM engine.

        Args:
            options: vLLM-specific runtime options
            accelerator_options: Hardware accelerator configuration
            artifacts_path: Path to cached model artifacts
            model_config: Model configuration (repo_id, revision, extra_config)
        """
        super().__init__(options, model_config=model_config)
        self.options: VllmVlmEngineOptions = options
        self.accelerator_options = accelerator_options
        self.artifacts_path = artifacts_path

        # These will be set during initialization
        self.device: Optional[str] = None
        self.llm: Any = None
        self.sampling_params: Any = None
        self.processor: Any = None

        # Initialize immediately if model_config is provided
        if self.model_config is not None:
            self.initialize()

    def initialize(self) -> None:
        """Initialize the vLLM engine."""
        if self._initialized:
            return

        _log.info("Initializing vLLM VLM inference engine...")

        try:
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
        except ImportError:
            if sys.version_info < (3, 14):
                raise ImportError(
                    "vLLM is not installed. Please install it via `pip install vllm` "
                    "to use vLLM for high-throughput VLM inference."
                )
            else:
                raise ImportError(
                    "vLLM is not installed. It is not yet available on Python 3.14."
                )

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

            _log.info(f"Loading vLLM model {repo_id} (revision: {revision})")

            # Resolve artifacts path
            from docling.models.utils.hf_model_download import (
                HuggingFaceModelDownloadMixin,
            )

            # Create a temporary mixin instance for downloading
            downloader = type(
                "Downloader",
                (HuggingFaceModelDownloadMixin,),
                {},
            )()

            # Wrapper to match expected signature
            def download_wrapper(repo_id: str, revision: str) -> Path:
                return downloader.download_models(repo_id, revision=revision)

            artifacts_path = resolve_model_artifacts_path(
                repo_id=repo_id,
                revision=revision,
                artifacts_path=self.artifacts_path,
                download_fn=download_wrapper,
            )

            # Split extra_generation_config into engine and sampling kwargs
            extra_cfg = self.model_config.extra_config
            load_cfg = {
                k: v for k, v in extra_cfg.items() if k in self._VLLM_ENGINE_KEYS
            }
            gen_cfg = {
                k: v for k, v in extra_cfg.items() if k in self._VLLM_SAMPLING_KEYS
            }

            unknown = sorted(
                k
                for k in extra_cfg.keys()
                if k not in self._VLLM_ENGINE_KEYS and k not in self._VLLM_SAMPLING_KEYS
            )
            if unknown:
                _log.warning("Ignoring unknown extra_config keys for vLLM: %s", unknown)

            # Construct LLM kwargs (engine/load-time)
            llm_kwargs: Dict[str, Any] = {
                "model": str(artifacts_path),
                "model_impl": "transformers",
                "limit_mm_per_prompt": {"image": 1},
                "revision": revision,
                "trust_remote_code": self.options.trust_remote_code,
                **load_cfg,
            }

            if self.device == "cpu":
                llm_kwargs.setdefault("enforce_eager", True)
            else:
                # Use configured gpu_memory_utilization or default
                llm_kwargs.setdefault(
                    "gpu_memory_utilization", self.options.gpu_memory_utilization
                )

            # Quantization support (if specified in extra_config)
            if "quantization" in extra_cfg:
                llm_kwargs.setdefault("quantization", extra_cfg["quantization"])

            # Initialize vLLM LLM
            self.llm = LLM(**llm_kwargs)

            # Initialize processor for prompt templating
            self.processor = AutoProcessor.from_pretrained(
                artifacts_path,
                trust_remote_code=self.options.trust_remote_code,
                revision=revision,
            )

            # Create default SamplingParams (will be overridden per-batch in predict_batch)
            # Use reasonable defaults since these come from input data
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=4096,
                **gen_cfg,
            )

            _log.info(f"Loaded vLLM model {repo_id} (revision: {revision})")

        self._initialized = True
        _log.info("vLLM runtime initialized")

    def predict_batch(self, input_batch: List[VlmEngineInput]) -> List[VlmEngineOutput]:
        """Run inference on a batch of inputs using vLLM.

        This method processes multiple images in a single batched vLLM call,
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
        if self.llm is None or self.processor is None or self.sampling_params is None:
            raise RuntimeError(
                "Model not loaded. Ensure EngineModelConfig was provided during initialization."
            )

        # Preprocess images
        images = preprocess_image_batch([inp.image for inp in input_batch])

        # Get prompt style from first input's extra config
        first_input = input_batch[0]
        prompt_style = first_input.extra_generation_config.get(
            "transformers_prompt_style",
            TransformersPromptStyle.CHAT,
        )

        # Format prompts
        prompts: List[Optional[str]] = []
        for input_data in input_batch:
            formatted_prompt = format_prompt_for_vlm(
                prompt=input_data.prompt,
                processor=self.processor,
                prompt_style=prompt_style,
                repo_id=self.model_config.repo_id if self.model_config else None,
            )
            prompts.append(formatted_prompt)

        # Build vLLM inputs
        llm_inputs = [
            {"prompt": p, "multi_modal_data": {"image": im}}
            for p, im in zip(prompts, images)
        ]

        # Update sampling params with input-specific settings
        from vllm import SamplingParams

        # Use first input's settings for the batch
        sampling_params = SamplingParams(
            temperature=first_input.temperature,
            max_tokens=first_input.max_new_tokens,
            stop=first_input.stop_strings or None,
            **{
                k: v
                for k, v in first_input.extra_generation_config.items()
                if k in self._VLLM_SAMPLING_KEYS
            },
        )

        # Generate
        start_time = time.time()
        outputs = self.llm.generate(llm_inputs, sampling_params=sampling_params)
        generation_time = time.time() - start_time

        _log.debug(
            f"vLLM generated {len(outputs)} outputs in {generation_time:.2f}s "
            f"({len(outputs) / generation_time:.1f} outputs/sec)"
        )

        # Create output objects
        results: List[VlmEngineOutput] = []
        for i, output in enumerate(outputs):
            text = output.outputs[0].text if output.outputs else ""
            stop_reason = (
                "end_of_sequence" if output.outputs[0].stop_reason else "length"
            )

            num_tokens = len(output.outputs[0].token_ids) if output.outputs else 0

            results.append(
                VlmEngineOutput(
                    text=text,
                    stop_reason=stop_reason,
                    metadata={
                        "generation_time": generation_time / len(input_batch),
                        "num_tokens": num_tokens,
                        "batch_size": len(input_batch),
                        "model": self.model_config.repo_id
                        if self.model_config
                        else "unknown",
                    },
                )
            )

        return results

    def cleanup(self) -> None:
        """Clean up vLLM resources."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        _log.info("vLLM runtime cleaned up")
