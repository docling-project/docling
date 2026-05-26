"""Top-level pytest conftest.

Auto-loads `.env` from the repository root so developers can keep secrets
(e.g. UPSTAGE_API_KEY for live OCR tests) out of the shell history and out of
git. The .env file itself is gitignored; see `.env.example` for the expected
schema.

Existing environment variables take precedence (override=False), so running
`UPSTAGE_API_KEY=... pytest` still works as before, and docker/k8s injected
env vars are never clobbered by a local .env.
"""
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a direct dep
    load_dotenv = None


def _load_repo_env() -> None:
    if load_dotenv is None:
        return
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


_load_repo_env()
