#!/bin/bash

set -e  # trigger failure on error - do not remove!
set -x  # display command on output

# Build each package into its own dist subdirectory so the PyPI publish
# action can upload them independently (otherwise a single `dist/` causes
# the second publish step to re-upload files and fail on `skip-existing: false`).

# Build docling-slim package (from repo root — source co-located)
echo "Building docling-slim package..."
uv build --out-dir dist/docling-slim

# Build docling package (meta-package, dependency-only wheel)
echo "Building docling package..."
# Backup placeholder README and copy root README for build
mv packages/docling/README.md packages/docling/README.md.placeholder
cp README.md packages/docling/README.md
(cd packages/docling && uv build --out-dir ../../dist/docling)
# Restore placeholder README
mv packages/docling/README.md.placeholder packages/docling/README.md

echo "Build complete."
echo "docling-slim artifacts:"
ls -lh dist/docling-slim/
echo "docling artifacts:"
ls -lh dist/docling/
