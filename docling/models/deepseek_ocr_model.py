"""DeepSeek OCR model integration for Docling.

DeepSeek-OCR is a Vision-Language Model (VLM) based OCR engine that uses
transformer models for document understanding and text extraction.
See: https://github.com/deepseek-ai/DeepSeek-OCR
"""

import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Type

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import DeepSeekOcrOptions, OcrOptions
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

# Model repository IDs
_DEEPSEEK_OCR_CUDA_REPO = "deepseek-ai/DeepSeek-OCR"
_DEEPSEEK_OCR_MPS_REPO = "Dogacel/DeepSeek-OCR-Metal-MPS"

# Minimum PyTorch version required for MPS support
# PyTorch 2.7.0+ is required for aten::_upsample_bicubic2d_aa operator on MPS
_MIN_PYTORCH_VERSION_MPS = (2, 7, 0)


def _parse_pytorch_version(version_str: str) -> tuple:
    """Parse PyTorch version string into a tuple of integers."""
    # Handle versions like "2.7.0", "2.7.0+cu118", "2.7.0.dev20241201"
    import re

    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if match:
        return tuple(int(x) for x in match.groups())
    return (0, 0, 0)


class DeepSeekOcrModel(BaseOcrModel):
    """DeepSeek OCR model for document text extraction.

    This model uses the DeepSeek-OCR Vision-Language Model to extract text
    from document images. Unlike traditional OCR engines that return bounding
    boxes, DeepSeek-OCR returns structured text/markdown output.

    Device Support:
    - CUDA (NVIDIA GPU): Optimal performance with flash_attention_2
    - MPS (Apple Silicon): Supported via MPS-compatible model fork (requires PyTorch 2.7.0+)
    - CPU: Not supported (model contains GPU-specific operations)
    """

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: DeepSeekOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: DeepSeekOcrOptions

        self.scale = 3  # multiplier for 72 dpi == 216 dpi

        # Device and dtype will be set during model initialization
        self.device: Any = None  # Will be set to torch.device during init
        self.dtype: Any = None  # Will be set to torch.dtype during init

        if self.enabled:
            self._init_model(accelerator_options, artifacts_path)

    def _init_model(
        self,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Path],
    ) -> None:
        """Initialize the DeepSeek-OCR model.

        Device selection priority:
        1. CUDA (NVIDIA GPU) - optimal performance with flash_attention_2
        2. MPS (Apple Silicon) - supported via MPS-compatible model (requires PyTorch 2.7.0+)
        3. CPU - not supported (raises error)
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "DeepSeek-OCR requires 'transformers' and 'torch' packages. "
                "Please install them via `pip install docling[deepseekocr]` to use this OCR engine. "
                "Alternatively, Docling has support for other OCR engines. See the documentation."
            )

        # Detect available devices
        has_cuda = torch.backends.cuda.is_built() and torch.cuda.is_available()
        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()

        # Determine device and configuration
        if has_cuda:
            # CUDA path - optimal performance
            self.device = torch.device("cuda")
            self.dtype = torch.bfloat16
            model_repo = (
                self.options.repo_id
            )  # Use configured repo (default: deepseek-ai/DeepSeek-OCR)

            # Determine attention implementation
            if self.options.attn_implementation:
                # User explicitly specified attention implementation
                attn_implementation = self.options.attn_implementation
            else:
                # Auto-detect: prefer flash_attention_2 if available, fallback to eager
                # Note: DeepSeek-OCR does not support 'sdpa', only 'flash_attention_2' or 'eager'
                try:
                    import flash_attn  # type: ignore[import-not-found]

                    attn_implementation = "flash_attention_2"
                    _log.debug("flash_attn package found, using flash_attention_2")
                except ImportError:
                    attn_implementation = "eager"
                    _log.info(
                        "flash_attn package not installed, using eager attention. "
                        "For optimal performance on CUDA, install flash-attn: "
                        "pip install flash-attn --no-build-isolation"
                    )

            _log.info(
                f"DeepSeek-OCR using CUDA device with bfloat16 precision and {attn_implementation} attention"
            )

        elif has_mps:
            # MPS path - Apple Silicon fallback
            self._validate_mps_pytorch_version(torch)
            self.device = torch.device("mps")
            self.dtype = torch.float16  # MPS requires float16, not bfloat16
            # Use MPS-compatible model unless user explicitly specified a different repo
            if self.options.repo_id == _DEEPSEEK_OCR_CUDA_REPO:
                model_repo = _DEEPSEEK_OCR_MPS_REPO
                _log.info(
                    f"DeepSeek-OCR: Switching to MPS-compatible model '{_DEEPSEEK_OCR_MPS_REPO}' "
                    "for Apple Silicon. To use a custom model, set repo_id in DeepSeekOcrOptions."
                )
            else:
                model_repo = self.options.repo_id
            # MPS requires eager attention (no flash_attention_2 support)
            attn_implementation = "eager"
            _log.info(
                "DeepSeek-OCR using MPS device (Apple Silicon) with float16 precision"
            )

        else:
            # No GPU available - raise helpful error
            raise RuntimeError(
                "DeepSeek-OCR requires a GPU (CUDA or Apple Silicon MPS). "
                "No compatible GPU was detected on this system.\n\n"
                "Options:\n"
                "  - For NVIDIA GPUs: Ensure CUDA is properly installed\n"
                "  - For Apple Silicon (M1/M2/M3/M4): Ensure PyTorch 2.7.0+ is installed\n"
                "  - For CPU-only environments: Use a different OCR engine:\n"
                "    * EasyOcrOptions - Good multilingual support\n"
                "    * TesseractOcrOptions - Classic OCR engine\n"
                "    * RapidOcrOptions - Fast CPU-based OCR"
            )

        # Validate attention implementation
        if attn_implementation == "sdpa":
            _log.warning(
                "DeepSeek-OCR does not support 'sdpa' attention. Falling back to 'eager'."
            )
            attn_implementation = "eager"

        _log.debug(f"DeepSeek-OCR using device: {self.device}, dtype: {self.dtype}")

        # Patch missing LlamaFlashAttention2 for compatibility with older/newer transformers
        self._patch_llama_flash_attention()

        # Determine model path (check for local cache)
        model_path = model_repo
        if artifacts_path is not None:
            repo_cache_folder = model_repo.replace("/", "--")
            local_path = artifacts_path / repo_cache_folder
            if local_path.exists():
                model_path = str(local_path)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.options.trust_remote_code,
        )
        # Set pad token to eos token (required for some models)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        # Note: We must use attn_implementation parameter (not _attn_implementation)
        # to properly override the model's config which may have flash_attention_2 hardcoded
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                trust_remote_code=self.options.trust_remote_code,
                use_safetensors=True,
                torch_dtype=self.dtype,
            )
            # Move model to device
            self.model = self.model.eval().to(self.device)

        except Exception as e:
            _log.error(f"Failed to load DeepSeek-OCR model: {e}")
            raise

    def _validate_mps_pytorch_version(self, torch) -> None:
        """Validate PyTorch version for MPS support.

        PyTorch 2.7.0+ is required for MPS because earlier versions lack the
        aten::_upsample_bicubic2d_aa operator needed by DeepSeek-OCR.
        """
        pytorch_version = _parse_pytorch_version(torch.__version__)
        if pytorch_version < _MIN_PYTORCH_VERSION_MPS:
            raise RuntimeError(
                f"DeepSeek-OCR on Apple Silicon MPS requires PyTorch {'.'.join(map(str, _MIN_PYTORCH_VERSION_MPS))} or later. "
                f"Current version: {torch.__version__}\n\n"
                "Please upgrade PyTorch:\n"
                "  pip install --upgrade torch>=2.7.0\n\n"
                "Or use a different OCR engine for older PyTorch versions:\n"
                "  * EasyOcrOptions - Supports MPS on older PyTorch\n"
                "  * TesseractOcrOptions - CPU-based, no PyTorch dependency"
            )

    def _patch_llama_flash_attention(self) -> None:
        """Patch missing LlamaFlashAttention2 for transformers compatibility.

        The DeepSeek-OCR model's custom code imports LlamaFlashAttention2 from
        transformers.models.llama.modeling_llama, but this class may not exist
        in all versions of transformers. This method adds a dummy class to
        prevent import errors.
        """
        try:
            from transformers.models.llama import modeling_llama

            if not hasattr(modeling_llama, "LlamaFlashAttention2"):
                # Create a dummy class that will raise an error if actually used
                class LlamaFlashAttention2Dummy:
                    def __init__(self, *args, **kwargs):
                        raise NotImplementedError(
                            "LlamaFlashAttention2 is not available in this version of transformers. "
                            "Please use attn_implementation='sdpa' or 'eager' instead of 'flash_attention_2', "
                            "or upgrade transformers to a version that supports flash attention."
                        )

                modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2Dummy  # type: ignore[attr-defined]
                _log.debug(
                    "Patched missing LlamaFlashAttention2 in transformers.models.llama.modeling_llama"
                )
        except Exception as e:
            _log.warning(f"Failed to patch LlamaFlashAttention2: {e}")

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells = []
                    for ocr_rect in ocr_rects:
                        # Skip zero area boxes
                        if ocr_rect.area() == 0:
                            continue

                        # Get high resolution image for the OCR region
                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )

                        # Run DeepSeek-OCR inference
                        cells = self._run_ocr(high_res_image, ocr_rect)
                        all_ocr_cells.extend(cells)

                        del high_res_image

                    # Post-process the cells
                    self.post_process_cells(all_ocr_cells, page)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    def _run_ocr(self, image, ocr_rect: BoundingBox) -> list[TextCell]:
        """Run DeepSeek-OCR on an image region and return TextCells.

        Since DeepSeek-OCR returns text without bounding boxes, we create
        a single TextCell covering the entire OCR region with the extracted text.
        """
        import tempfile

        tmp_path = None
        tmp_dir = None
        try:
            # Create a temporary directory for output (required by DeepSeek-OCR)
            tmp_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")

            # Save image to temporary file (DeepSeek-OCR expects file path)
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, mode="wb", dir=tmp_dir
            ) as tmp_file:
                tmp_path = tmp_file.name
                image.save(tmp_path, format="PNG")

            _log.debug(f"DeepSeek-OCR processing image: {tmp_path}")

            # Verify the file exists and has content
            if not Path(tmp_path).exists():
                _log.warning(f"Temp file was not created: {tmp_path}")
                return []

            # Run inference using DeepSeek-OCR's infer method
            # Note: output_path is required even if save_results=False
            # eval_mode=True is required to get the text returned instead of just printed
            infer_kwargs: dict[str, Any] = {
                "prompt": self.options.prompt,
                "image_file": tmp_path,
                "output_path": tmp_dir,
                "base_size": self.options.base_size,
                "image_size": self.options.image_size,
                "crop_mode": self.options.crop_mode,
                "save_results": False,
                "test_compress": False,
                "eval_mode": True,
            }

            # Only pass device/dtype for MPS model (not supported by original CUDA model)
            # The MPS-compatible fork (Dogacel/DeepSeek-OCR-Metal-MPS) added these parameters
            if self.device.type == "mps":
                infer_kwargs["device"] = self.device
                infer_kwargs["dtype"] = self.dtype

            result = self.model.infer(self.tokenizer, **infer_kwargs)

            # Parse the result text into lines
            if result and isinstance(result, str) and result.strip():
                cells = self._parse_text_to_cells(result, ocr_rect)
                return cells

            return []

        except Exception as e:
            _log.warning(f"DeepSeek-OCR inference failed: {e}")
            return []
        finally:
            # Clean up temp files and directory
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
            if tmp_dir:
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _parse_text_to_cells(self, text: str, ocr_rect: BoundingBox) -> list[TextCell]:
        """Parse extracted text into TextCell objects.

        Since DeepSeek-OCR doesn't provide bounding boxes, we create cells
        based on text lines, distributing them evenly across the OCR region.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if not lines:
            return []

        # Calculate approximate line height
        region_height = ocr_rect.b - ocr_rect.t
        line_height = region_height / len(lines) if lines else region_height

        cells = []
        for idx, line in enumerate(lines):
            # Calculate approximate position for this line
            top = ocr_rect.t + (idx * line_height)
            bottom = top + line_height

            cell = TextCell(
                index=idx,
                text=line,
                orig=line,
                from_ocr=True,
                confidence=1.0,  # DeepSeek-OCR doesn't provide confidence scores
                rect=BoundingRectangle.from_bounding_box(
                    BoundingBox.from_tuple(
                        coord=(
                            ocr_rect.l / self.scale,
                            top / self.scale,
                            ocr_rect.r / self.scale,
                            bottom / self.scale,
                        ),
                        origin=CoordOrigin.TOPLEFT,
                    )
                ),
            )
            cells.append(cell)

        return cells

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return DeepSeekOcrOptions
