import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    HuggingFaceVlmOptions,
)
from docling.models.base_model import BasePageModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class HuggingFaceVlmModel:
    @staticmethod
    def download_models(
        repo_id: str,
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        if not progress:
            disable_progress_bars()
        download_path = snapshot_download(
            repo_id=repo_id,
            force_download=force,
            local_dir=local_dir,
            # revision="v0.0.1",
        )

        return Path(download_path)
