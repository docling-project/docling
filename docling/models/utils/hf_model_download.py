# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

# A fetch faster than this was served from the local cache. Downloading even a
# small model over the network takes longer, so the split reads correctly in the
# log without asking huggingface_hub whether every file was already present.
_CACHE_HIT_SECONDS = 1.0

# A full 40-character lowercase hex commit SHA always resolves to the same
# repository tree. Any other revision (None, which resolves to ``main``, a
# branch, or a tag) is a moving ref whose contents can change.
_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")

# This module sits in tach's ``foundation`` layer, below ``docling.datamodel``,
# so the refusal toggle is read straight from the environment rather than
# through ``docling.datamodel.settings``. Read at call time, not import time, so
# the variable can be set after import.
_REFUSE_ENV_VAR = "DOCLING_SECURITY_REFUSE_UNPINNED_REMOTE_CODE"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def _refuse_unpinned_remote_code() -> bool:
    """Return True when the env var asks for a hard refusal rather than a warning."""
    return os.environ.get(_REFUSE_ENV_VAR, "").strip().lower() in _TRUTHY_ENV_VALUES


def is_pinned_revision(revision: Optional[str]) -> bool:
    """Check whether ``revision`` is a full commit SHA.

    The value is matched exactly as it is passed to the Hugging Face Hub, so
    surrounding whitespace or uppercase hex digits do not count as pinned.

    Args:
        revision: The revision passed to the Hugging Face Hub.

    Returns:
        True if ``revision`` is a 40-character lowercase hex commit SHA; False for
        ``None``, branches, tags, short SHAs and any other value.
    """
    if not revision:
        return False
    return _COMMIT_SHA_RE.fullmatch(revision) is not None


def warn_on_unpinned_trust_remote_code(
    repo_id: str,
    revision: Optional[str],
    trust_remote_code: bool,
) -> None:
    """Flag a ``trust_remote_code`` model download from an unpinned revision.

    With ``trust_remote_code=True``, loading the model executes Python code from
    the repository, so the revision must be a commit SHA for the executed code to
    be fixed. ``download_hf_model`` calls this before fetching. An unpinned
    revision logs a warning, or raises when
    ``DOCLING_SECURITY_REFUSE_UNPINNED_REMOTE_CODE`` is set to ``1``, ``true``,
    ``yes`` or ``on`` (case-insensitive).

    Args:
        repo_id: The Hugging Face repository ID.
        revision: The requested revision; ``None`` resolves to ``main``.
        trust_remote_code: Whether the model is loaded with remote code enabled.

    Raises:
        ValueError: If ``trust_remote_code`` is True, ``revision`` is not a commit
            SHA, and the refusal environment variable is set to a truthy value.
    """
    if not trust_remote_code or is_pinned_revision(revision):
        return

    message = (
        "SECURITY: model repo '{repo}' is loaded with trust_remote_code=True at "
        "the moving revision '{rev}'. Custom code from this repository is executed "
        "on load, and a moving ref can be changed at any time (force-push or Hub "
        "account compromise) to run attacker-controlled code. Pin 'revision' to a "
        "full 40-character commit SHA from a reviewed version of the repo. Set "
        "DOCLING_SECURITY_REFUSE_UNPINNED_REMOTE_CODE=1 to refuse instead of warn."
    ).format(repo=repo_id, rev=revision or "main")

    if _refuse_unpinned_remote_code():
        raise ValueError(message)
    _log.warning(message)


def download_hf_model(
    repo_id: str,
    local_dir: Optional[Path] = None,
    force: bool = False,
    progress: bool = False,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    # Fires before anything is fetched, so a refusal blocks the download and a
    # warning precedes it in the log.
    warn_on_unpinned_trust_remote_code(repo_id, revision, trust_remote_code)

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
        trust_remote_code: bool = False,
    ) -> Path:
        return download_hf_model(
            repo_id=repo_id,
            local_dir=local_dir,
            force=force,
            progress=progress,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
