# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import re
import time
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

# A fetch faster than this was served from the local cache. Downloading even a
# small model over the network takes longer, so the split reads correctly in the
# log without asking huggingface_hub whether every file was already present.
_CACHE_HIT_SECONDS = 1.0

# A Hugging Face revision that is a full 40-character hex commit SHA is
# immutable: it always resolves to the exact same tree. Anything else -- None
# (defaults to ``main``), a branch, or a tag -- is a *moving* ref whose contents
# can change under us after review (a force-push, a retag, or a compromised Hub
# account). That distinction is the whole basis of the supply-chain guard below.
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def is_pinned_revision(revision: Optional[str]) -> bool:
    """Return True only if ``revision`` is a full 40-hex-char commit SHA.

    Branches, tags, ``None`` (which resolves to ``main``) and short SHAs are all
    treated as *moving* refs and return False.
    """
    if not revision:
        return False
    return _COMMIT_SHA_RE.match(revision.strip().lower()) is not None


def warn_on_unpinned_trust_remote_code(
    repo_id: str,
    revision: Optional[str],
    trust_remote_code: bool,
) -> None:
    """Guard the trust_remote_code + moving-revision supply-chain risk.

    When a model repo is loaded with ``trust_remote_code=True`` (custom Python
    from the Hub is executed on load) *and* the revision is a moving ref rather
    than a pinned commit SHA, an attacker who force-pushes or takes over that Hub
    repo gains arbitrary code execution on the next model load. This helper is the
    single chokepoint (called from :func:`download_hf_model`) that flags it.

    Default behaviour is a prominent ``log.warning``. Set
    ``DOCLING_SECURITY_REFUSE_UNPINNED_REMOTE_CODE=1`` (settings
    ``security.refuse_unpinned_remote_code``) to raise instead, for callers that
    want to fail closed.
    """
    if not trust_remote_code or is_pinned_revision(revision):
        return

    # Imported lazily to keep this module import-cheap and cycle-free.
    from docling.datamodel.settings import settings

    message = (
        "SECURITY: model repo '{repo}' is loaded with trust_remote_code=True at "
        "the moving revision '{rev}'. Custom code from this repository is executed "
        "on load, and a moving ref can be changed at any time (force-push or Hub "
        "account compromise) to run attacker-controlled code. Pin 'revision' to a "
        "full 40-character commit SHA from a reviewed version of the repo. Set "
        "DOCLING_SECURITY_REFUSE_UNPINNED_REMOTE_CODE=1 to refuse instead of warn."
    ).format(repo=repo_id, rev=revision or "main")

    if settings.security.refuse_unpinned_remote_code:
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
