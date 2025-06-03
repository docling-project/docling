import logging
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


def download_hf_model(
    repo_id: str,
    local_dir: Optional[Path] = None,
    force: bool = False,
    progress: bool = False,
    revision: Optional[str] = None,
) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    if not progress:
        disable_progress_bars()
    download_path = snapshot_download(
        repo_id=repo_id,
        force_download=force,
        local_dir=local_dir,
        revision=revision,
    )

    return Path(download_path)


class HuggingFaceModelDownloadMixin:
    @staticmethod
    def download_models(
        repo_id: str,
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        return download_hf_model(
            repo_id=repo_id, local_dir=local_dir, force=force, progress=progress
        )
