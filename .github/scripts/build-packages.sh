#!/bin/bash

set -e  # trigger failure on error - do not remove!
set -x  # display command on output

# Create dist directory if it doesn't exist
mkdir -p dist

# Build docling-slim package (from repo root — source co-located)
echo "Building docling-slim package..."
uv build --out-dir dist

# Build docling package (meta-package, dependency-only wheel)
echo "Building docling package..."
(cd packages/docling && uv build --out-dir ../../dist)

echo "Build complete. Packages are in dist/"
ls -lh dist/
