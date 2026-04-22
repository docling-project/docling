# Default: list available recipes
default:
    @just --list

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Install the CI-style development environment
setup:
    uv sync --frozen --group dev --all-extras --no-group docs --no-group examples

# Install git hooks via prek
hooks-install:
    uv run prek install

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Run the full local validation suite on all files
check:
    just validate-all

# Run hooks on staged files only
validate:
    uv run prek run

# Run hooks on all files in the repo
validate-all:
    uv run prek run --all-files

# Run Ruff and dprint auto-format/fixers
fix:
    uv run ruff format --config=pyproject.toml docling tests docs/examples
    uv run ruff check --fix --config=pyproject.toml docling tests docs/examples
    uv run dprint fmt

# Run MyPy
typecheck:
    uv run --no-sync mypy docling

# Run Tach module-boundary checks
tach:
    uv run --no-sync tach check

# Check dprint formatting without modifying files
dprint-check:
    uv run --no-sync dprint check

# Apply dprint formatting for configured files
dprint-fix:
    uv run --no-sync dprint fmt

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Run the default test suite
test:
    uv run pytest -v
