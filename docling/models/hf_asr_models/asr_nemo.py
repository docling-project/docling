import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import AsrPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    HuggingFaceAsrOptions,
)
from docling.models.base_model import BasePageModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class AsrNemoModel(BasePageModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        asr_options: HuggingFaceAsrOptions,
    ):
        self.enabled = enabled

        self.asr_options = asr_options

        if self.enabled:
            import nemo.collections.asr as nemo_asr

            device = decide_device(accelerator_options.device)
            self.device = device

            _log.debug(f"Available device for HuggingFace ASR: {device}")

            repo_cache_folder = asr_options.repo_id.replace("/", "--")

            # PARAMETERS:
            if artifacts_path is None:
                artifacts_path = self.download_models(self.asr_options.repo_id)
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            self.model = nemo_asr.models.ASRModel.from_pretrained(
                "nvidia/parakeet-tdt-0.6b-v2"
            )
