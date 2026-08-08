import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import disable_progress_bars

from docling.exceptions import DoclingModelDownloadError

_log = logging.getLogger(__name__)


def download_hf_model(
    repo_id: str,
    local_dir: Optional[Path] = None,
    force: bool = False,
    progress: bool = False,
    revision: Optional[str] = None,
    token: Optional[str | bool] = None,
) -> Path:
    """Download model files from a HuggingFace repo.

    Args:
        repo_id (`str`):
            HuggingFace repo. User or organization name.
        local_dir (`Path`, *optional*, defaults to `None`):
            If provided, downloaded files go in this directory. Default is `None`.
        force (`bool`, defaults to `False`):
            Whether to force a download even if the file exists in a local cache. Default is `False`.
        progress (`bool`, defaults to `False`):
            Toggles progress bars. Default is `False`.
        revision (`str`, *optional*, defaults to `None`):
            If provided, an optional Git revision id. Can be a branch name, tag, or commit hash. Default is `None`.
        token (`str`, `bool`, *optional*, defaults to `None`):
            If provided, the token is used to authenticate and accelerate downloads. Default is `None`.
                - If `True` OR None, the token is read from the `HF_TOKEN` environment variable.
                - If `False`, the token is explicitly ignored and only unauthenticated requests are made.
                - If a string, used directly as the authentication token.
    Returns:
        Path object. Path to where the downloaded model files are located.

    Raises:
        DoclingModelDownloadError: Invalid token, unauthorized token, general download error.
    """
    if not progress:
        disable_progress_bars()
    try:
        download_path = snapshot_download(
            repo_id=repo_id,
            force_download=force,
            local_dir=local_dir,
            revision=revision,
            token=token,
        )
        return Path(download_path)

    except HfHubHTTPError as e:
        # Check if it was an auth failure
        if e.response is not None and e.response.status_code == 401:
            raise DoclingModelDownloadError(
                f"Authentication failed: The provided HuggingFace token is invalid or unauthorized "
                f"for repository '{repo_id}'.",
                original_exception=e,
            ) from e

        # General network/repository issues
        raise DoclingModelDownloadError(
            f"Failed to download HuggingFace model repository '{repo_id}': {e}",
            original_exception=e,
        ) from e

    except Exception as e:
        # Fallback for unexpected local filesystem errors or library failures
        raise DoclingModelDownloadError(
            f"An unexpected error occurred while downloading model '{repo_id}': {e}",
            original_exception=e,
        ) from e


class HuggingFaceModelDownloadMixin:
    @staticmethod
    def download_models(
        repo_id: str,
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
        revision: Optional[str] = None,
        hf_token: Optional[str | bool] = None,
    ) -> Path:
        """Download model files from a HuggingFace repo.

        Args:
            repo_id (`str`):
                HuggingFace repo. User or organization name.
            local_dir (`Path`, *optional*, defaults to `None`):
                If provided, downloaded files go in this directory. Default is `None`.
            force (`bool`, defaults to `False`):
                Whether to force a download even if the file exists in a local cache. Default is `False`.
            progress (`bool`, defaults to `False`):
                Toggles progress bars. Default is `False`.
            revision (`str`, *optional*, defaults to `None`):
                If provided, an optional Git revision id. Can be a branch name, tag, or commit hash. Default is `None`.
            token (`str`, `bool`, *optional*, defaults to `None`):
                If provided, the token is used to authenticate and accelerate downloads. Default is `None`.
                    - If `True` OR None, the token is read from the `HF_TOKEN` environment variable.
                    - If `False`, the token is explicitly ignored and only unauthenticated requests are made.
                    - If a string, used directly as the authentication token.
        Returns:
            Path object. Path to where the downloaded model files are located.

        Raises:
            DoclingModelDownloadError: Invalid token, unauthorized token, general download error.
        """
        return download_hf_model(
            repo_id=repo_id,
            local_dir=local_dir,
            force=force,
            progress=progress,
            revision=revision,
            token=hf_token,
        )
