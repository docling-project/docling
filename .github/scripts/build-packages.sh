#!/bin/bash

set -e  # trigger failure on error - do not remove!
set -x  # display command on output

# Create dist directory if it doesn't exist
mkdir -p dist

# Build docling-slim package
echo "Building docling-slim package..."
(cd packages/docling-slim && uv build --out-dir ../../dist)

# Build docling package  
echo "Building docling package..."
(cd packages/docling && uv build --out-dir ../../dist)

echo "Build complete. Packages are in dist/"
ls -lh dist/
