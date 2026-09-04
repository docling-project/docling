# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import time
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

# A fetch faster than this was served from the local cache. Downloading even a
# small model over the network takes longer, so the split reads correctly in the
# log without asking huggingface_hub whether every file was already present.
_CACHE_HIT_SECONDS = 1.0


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

    # Progress bars are off by default, so without these lines a multi-gigabyte
    # download is indistinguishable from a hang.
    _log.info("Fetching model %s (revision: %s)...", repo_id, revision or "main")
    start_time = time.monotonic()
    download_path = snapshot_download(
        repo_id=repo_id,
        force_download=force,
        local_dir=local_dir,
        revision=revision,
    )
    elapsed = time.monotonic() - start_time

    if elapsed < _CACHE_HIT_SECONDS:
        _log.info("Model %s already cached at %s", repo_id, download_path)
    else:
        _log.info(
            "Downloaded model %s to %s in %.2f sec.", repo_id, download_path, elapsed
        )

    return Path(download_path)


class HuggingFaceModelDownloadMixin:
    @staticmethod
    def download_models(
        repo_id: str,
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
        revision: Optional[str] = None,
    ) -> Path:
        return download_hf_model(
            repo_id=repo_id,
            local_dir=local_dir,
            force=force,
            progress=progress,
            revision=revision,
        )
