#!/bin/bash

set -e  # trigger failure on error - do not remove!
set -x  # display command on output

if [ -z "${TARGET_VERSION}" ]; then
    >&2 echo "No TARGET_VERSION specified"
    exit 1
fi
CHGLOG_FILE="${CHGLOG_FILE:-CHANGELOG.md}"

# update package versions:
#   - root pyproject.toml           = docling-slim
#   - packages/docling/pyproject.toml        = docling (meta-package)
#   - packages/docling-client/pyproject.toml = docling-client (meta-package)
uvx --from=toml-cli toml set --toml-path=pyproject.toml project.version "${TARGET_VERSION}"
uvx --from=toml-cli toml set --toml-path=packages/docling/pyproject.toml project.version "${TARGET_VERSION}"
uvx --from=toml-cli toml set --toml-path=packages/docling-client/pyproject.toml project.version "${TARGET_VERSION}"

# Update all docling-slim dependencies in docling and docling-client packages using Python
TARGET_VERSION="${TARGET_VERSION}" python3 << 'PYTHON_SCRIPT'
import os
import re
from pathlib import Path

target_version = os.environ['TARGET_VERSION']

# Pattern to match docling-slim dependencies with version pinning
# Matches: docling-slim[extra]==version or docling-slim==version
pattern = r'(docling-slim(?:\[[^\]]+\])?)==[\d\.]+'

for pyproject_path in [
    Path("packages/docling/pyproject.toml"),
    Path("packages/docling-client/pyproject.toml"),
]:
    content = pyproject_path.read_text()
    updated_content = re.sub(pattern, rf'\1=={target_version}', content)
    pyproject_path.write_text(updated_content)
    print(f"Updated all docling-slim dependencies in {pyproject_path} to version {target_version}")
PYTHON_SCRIPT

UV_FROZEN=0 uv lock --upgrade-package docling --upgrade-package docling-slim --upgrade-package docling-client

# collect release notes
REL_NOTES=$(mktemp)
uv run --no-sync semantic-release changelog --unreleased >> "${REL_NOTES}"

# update changelog
TMP_CHGLOG=$(mktemp)
TARGET_TAG_NAME="v${TARGET_VERSION}"
RELEASE_URL="$(gh repo view --json url -q ".url")/releases/tag/${TARGET_TAG_NAME}"
printf "## [${TARGET_TAG_NAME}](${RELEASE_URL}) - $(date -Idate)\n\n" >> "${TMP_CHGLOG}"
cat "${REL_NOTES}" >> "${TMP_CHGLOG}"
if [ -f "${CHGLOG_FILE}" ]; then
    printf "\n" | cat - "${CHGLOG_FILE}" >> "${TMP_CHGLOG}"
fi
mv "${TMP_CHGLOG}" "${CHGLOG_FILE}"

# push changes
git config --global user.name 'github-actions[bot]'
git config --global user.email 'github-actions[bot]@users.noreply.github.com'
git add pyproject.toml packages/docling/pyproject.toml packages/docling-client/pyproject.toml uv.lock "${CHGLOG_FILE}"
COMMIT_MSG="chore: bump version to ${TARGET_VERSION} [skip ci]"
git commit -m "${COMMIT_MSG}"
git push origin main

# create GitHub release (incl. Git tag)
gh release create "${TARGET_TAG_NAME}" -F "${REL_NOTES}"
