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

# Run read-only local checks
check:
    just check-all

# Run read-only local checks on all supported files
check-all:
    uv run ruff format --check --config=pyproject.toml docling tests docs/examples
    uv run ruff check --config=pyproject.toml docling tests docs/examples
    uv run --no-sync ty check
    uv run --no-sync tach check
    python3 scripts/check_tach_module_coverage.py
    python3 scripts/check_max_lines.py
    uv run --no-sync dprint check
    uv lock --locked

# Run hooks on the current changeset; hooks may modify those files
validate:
    @files="$( \
        { \
            git diff --name-only --diff-filter=ACMR; \
            git diff --cached --name-only --diff-filter=ACMR; \
            git ls-files --others --exclude-standard; \
        } | sort -u \
    )"; \
    if [ -z "$files" ]; then \
        echo "No changed files to validate."; \
    else \
        printf '%s\n' "$files" | xargs uv run prek run --files; \
    fi

# Run hooks on all files in the repo; hooks may modify files
validate-all:
    uv run prek run --all-files

# Run Ruff and dprint auto-format/fixers
fix:
    uv run ruff format --config=pyproject.toml docling tests docs/examples
    uv run ruff check --fix --config=pyproject.toml docling tests docs/examples
    uv run dprint fmt

# Run ty
typecheck:
    uv run --no-sync ty check

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
