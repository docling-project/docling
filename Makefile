.PHONY: help setup hooks-install check check-all validate validate-all fix typecheck tach dprint-check dprint-fix test

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; print "Available targets:"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install CI-style development environment.
	uv sync --frozen --group dev --all-extras --no-group docs --no-group examples

hooks-install: ## Install git hooks via prek.
	uv run prek install

check: check-all ## Run read-only local checks.

check-all: ## Run all read-only local checks.
	uv run ruff format --check --config=pyproject.toml docling tests docs/examples
	uv run ruff check --config=pyproject.toml docling tests docs/examples
	uv run --no-sync ty check
	uv run --no-sync tach check
	python3 scripts/check_tach_module_coverage.py
	python3 scripts/check_max_lines.py
	uv run --no-sync dprint check --config .github/dprint.json --config-discovery=false
	uv lock --locked

validate: ## Run hooks on the current changeset.
	@files="$$( \
		{ \
			git diff --name-only --diff-filter=ACMR; \
			git diff --cached --name-only --diff-filter=ACMR; \
			git ls-files --others --exclude-standard; \
		} | sort -u \
	)"; \
	if [ -z "$$files" ]; then \
		echo "No changed files to validate."; \
	else \
		printf '%s\n' "$$files" | xargs uv run prek run --files; \
	fi

validate-all: ## Run hooks on all files.
	uv run prek run --all-files

fix: ## Run Ruff and dprint auto-format/fixers.
	uv run ruff format --config=pyproject.toml docling tests docs/examples
	uv run ruff check --fix --config=pyproject.toml docling tests docs/examples
	uv run dprint fmt --config .github/dprint.json --config-discovery=false

typecheck: ## Run ty.
	uv run --no-sync ty check

tach: ## Run Tach module-boundary checks.
	uv run --no-sync tach check

dprint-check: ## Run dprint in check mode.
	uv run --no-sync dprint check --config .github/dprint.json --config-discovery=false

dprint-fix: ## Run dprint formatter.
	uv run --no-sync dprint fmt --config .github/dprint.json --config-discovery=false

test: ## Run the default test suite.
	uv run pytest -v
