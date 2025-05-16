import logging
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


class HuggingFaceVlmModel:

    @staticmethod
    def map_device_to_cpu_if_mlx(device: str) -> str:
        if device == "mps":
            _log.warning(
                "Mapping mlx to cpu for AutoModelForCausalLM, use MLX framework!"
            )
            return "cpu"

        return device
        
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
