# Docling Slim Refactoring Plan - v2.85.0

## Project Context
- **Current Package**: `docling` (version 2.85.0)
- **Goal**: Split into `docling-slim` (minimal dependencies) and `docling` (current default dependencies)
- **Repository**: https://github.com/docling-project/docling

## Key Requirements

1. **No source code movement** - Source stays in `docling/`, only metadata moves
2. **uv workspace structure** - Real package directories with standard workspace setup
3. **docling depends on docling-slim** - Same dependencies as current docling
4. **Exact version pinning** - docling depends on exact version of docling-slim
5. **CLI in docling only** - Scripts in full package, not in slim
6. **Fine-grained extras** - Split components into granular options
7. **CI/CD pushes both packages** - Build from package directories
8. **Local development** - Editable installs via workspace

## Repository Structure

```
docling/  (current repository)
├── pyproject.toml                    # Workspace root configuration
├── packages/
│   ├── docling-slim/
│   │   └── pyproject.toml           # docling-slim package metadata
│   └── docling/
│       └── pyproject.toml           # docling package metadata (depends on slim)
├── docling/                          # Source code (UNCHANGED LOCATION)
│   ├── __init__.py
│   ├── document_converter.py
│   ├── cli/                         # CLI code stays here
│   ├── datamodel/                   # Data models
│   └── ...
├── tests/                            # Tests (UNCHANGED)
├── docs/                             # Docs (UNCHANGED)
└── .github/
    └── workflows/
        ├── ci.yml                    # Modified: test both packages
        └── pypi.yml                  # Modified: publish both (slim first, then full)
```

**Key Points:**
- Source code stays in `docling/` at repository root
- Both packages reference the same source via `packages = ["docling"]` in their pyproject.toml
- Only package metadata files are in `packages/` subdirectories
- Standard uv workspace pattern with shared source code

## Dependency Categorization

### **docling-slim BASE (8 packages)** - ~50MB, Library-First

```python
dependencies = [
    'pydantic>=2.0.0,<3.0.0',
    'docling-core>=2.70.0,<3.0.0',
    'pydantic-settings>=2.3.0,<3.0.0',
    'filetype>=1.2.0,<2.0.0',
    'requests>=2.32.2,<3.0.0',
    'certifi>=2024.7.4',
    'pluggy>=1.0.0,<2.0.0',
    'tqdm>=4.65.0,<5.0.0',
]
```

**Provides:**
- Core data models (DoclingDocument, ConversionResult)
- Document format definitions
- Basic I/O utilities
- Library-only (no CLI tools)

### **docling-slim EXTRAS (Fine-grained)**

#### **1. PDF Backend Options (Fine-grained by backend)**

**`[backend-pypdfium2]` - PyPdfium2 backend (basic PDF parsing)**
```python
'pypdfium2>=4.30.0,!=4.30.1,<6.0.0',
'numpy>=1.24.0,<3.0.0',
'pillow>=10.0.0,<13.0.0',
```

**`[backend-docling-parse]` - Docling Parse backend (advanced PDF parsing)**
```python
'docling-parse>=5.3.2,<6.0.0',
'pypdfium2>=4.30.0,!=4.30.1,<6.0.0',
'numpy>=1.24.0,<3.0.0',
'pillow>=10.0.0,<13.0.0',
```

**`[parse]` - Convenience: Complete parsing with docling-parse backend**
```python
'docling-slim[backend-docling-parse]',
```

#### **2. Model Dependencies (Fine-grained)**

**`[models-core]` - Core dependencies for models (2 packages)**
```python
'scipy>=1.6.0,<2.0.0',  # Mathematical operations
'rtree>=1.3.0,<2.0.0',  # Spatial indexing (if not already from parse-spatial)
```

**`[models-inference]` - ML model execution (6 packages, ~2GB)**
```python
'torch>=2.2.2,<3.0.0',
'torchvision>=0,<1',
'docling-ibm-models>=3.13.0,<4',
'accelerate>=1.0.0,<2',
'huggingface_hub>=0.23,<2',
'defusedxml>=0.7.1,<0.8.0',
```

**`[models]` - Convenience: Complete model support**
```python
'docling-slim[parse,models-core,models-inference]',
```

#### **3. OCR Engines (Separate extras for each)**

**`[ocr-rapidocr]` - Basic RapidOCR**
```python
'rapidocr>=3.3,<4.0.0',
```

**`[ocr-rapidocr-onnx]` - RapidOCR with ONNX runtime**
```python
'rapidocr>=3.3,<4.0.0',
'onnxruntime>=1.7.0,<2.0.0 ; python_version < "3.14"',
```

**`[ocr-easyocr]` - EasyOCR engine**
```python
'easyocr>=1.7,<2.0',
```

**`[ocr-tesserocr]` - Tesseract with pandas**
```python
'tesserocr>=2.7.1,<3.0.0',
'pandas>=2.1.4,<4.0.0',
```

**`[ocr-mac]` - macOS native OCR**
```python
'ocrmac>=1.0.0,<2.0.0 ; sys_platform == "darwin"',
```

#### **4. Input Format Support (Fine-grained)**

**`[format-docx]` - Word documents**
```python
'python-docx>=1.1.2,<2.0.0',
```

**`[format-pptx]` - PowerPoint presentations**
```python
'python-pptx>=1.0.2,<2.0.0',
```

**`[format-xlsx]` - Excel spreadsheets**
```python
'openpyxl>=3.1.5,<4.0.0',
```

**`[format-office]` - Convenience: All Office formats**
```python
'docling-slim[format-docx,format-pptx,format-xlsx]',
```

**`[format-html]` - HTML parsing**
```python
'beautifulsoup4>=4.12.3,<5.0.0',
'lxml>=4.0.0,<7.0.0',
```

**`[format-markdown]` - Markdown parsing**
```python
'marko>=2.1.2,<3.0.0',
```

**`[format-web]` - Convenience: HTML + Markdown**
```python
'docling-slim[format-html,format-markdown]',
```

**`[format-latex]` - LaTeX documents**
```python
'pylatexenc>=2.10,<3.0',
```

**`[format-xbrl]` - Financial reports (XBRL)**
```python
'arelle-release>=2.38.17,<3.0.0',
```

#### **5. Advanced Features**

**`[vlm]` - Vision Language Models**
```python
'transformers>=4.42.0,<6.0.0,!=5.0.*,!=5.1.*,!=5.2.*,!=5.3.*',
'accelerate>=1.2.1,<2.0.0',
'mlx-vlm>=0.3.0,<1.0.0 ; python_version >= "3.10" and sys_platform == "darwin" and platform_machine == "arm64"',
'qwen-vl-utils>=0.0.11',
```

**`[asr]` - Automatic Speech Recognition**
```python
'mlx-whisper>=0.4.3 ; python_version >= "3.10" and sys_platform == "darwin" and platform_machine == "arm64"',
'openai-whisper>=20250625',
'numba>=0.63.0',
```

**`[htmlrender]` - HTML rendering with Playwright**
```python
'playwright>=1.58.0',
```

**`[remote-serving]` - Remote model inference**
```python
'tritonclient[grpc]>=2.65.0,<3.0.0',
```

**`[onnxruntime]` - ONNX runtime variants**
```python
'onnxruntime<1.24 ; python_version < "3.14" and sys_platform == "darwin"',
'onnxruntime-gpu<1.24 ; python_version < "3.14" and (sys_platform == "linux" or sys_platform == "win32")',
```

**`[cli]` - Command-line interface**
```python
'typer>=0.12.5,<0.22.0',
```

**`[chunking]` - Document chunking**
```python
'docling-core[chunking]>=2.70.0,<3.0.0',
```

**`[extraction]` - Information extraction functionality**
```python
'polyfactory>=2.22.2',
```

#### **6. Convenience Extras**

**`[standard]` - Standard installation (matches current docling default)**
```python
'docling-slim[parse,models,ocr-rapidocr,format-office,format-web,latex,chunking,polyfactory]',
```

**`[all]` - Everything**
```python
'docling-slim[standard,vlm,asr,htmlrender,xbrl,remote-serving,onnxruntime,ocr-easyocr,ocr-tesserocr,ocr-mac]',
```

## docling Package Dependencies

The `docling` package depends on `docling-slim[standard]` plus CLI support:

```python
dependencies = [
    'docling-slim[standard]==2.85.0',
    'typer>=0.12.5,<0.22.0',  # CLI dependency
]
```

## Detailed pyproject.toml Structures

### Root pyproject.toml (Workspace Configuration)

```toml
[project]
name = "docling-workspace"
version = "2.85.0"
description = "Docling workspace - do not install directly"
requires-python = ">=3.10,<4.0"

# Minimal dependencies for workspace management
dependencies = []

[tool.uv.workspace]
members = ["packages/docling-slim", "packages/docling"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Note:** This root pyproject.toml is for workspace management only. Users install `docling-slim` or `docling` packages.

### packages/docling-slim/pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "docling-slim"
version = "2.85.0"  # DO NOT EDIT, updated automatically
description = "Lightweight SDK for parsing documents (minimal dependencies, opt-in extras)"

# Point to shared source code at repository root
packages = ["docling"]
license = "MIT"
keywords = [
  "docling",
  "convert",
  "document",
  "pdf",
  "docx",
  "html",
  "markdown",
  "layout model",
  "segmentation",
]
classifiers = [
  "Operating System :: MacOS :: MacOS X",
  "Operating System :: POSIX :: Linux",
  "Operating System :: Microsoft :: Windows",
  "Development Status :: 5 - Production/Stable",
  "Intended Audience :: Developers",
  "Intended Audience :: Science/Research",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Programming Language :: Python :: 3.14",
]
readme = "README.md"
authors = [
  { name = "Christoph Auer", email = "cau@zurich.ibm.com" },
  { name = "Michele Dolfi", email = "dol@zurich.ibm.com" },
  { name = "Maxim Lysak", email = "mly@zurich.ibm.com" },
  { name = "Nikos Livathinos", email = "nli@zurich.ibm.com" },
  { name = "Ahmed Nassar", email = "ahn@zurich.ibm.com" },
  { name = "Panos Vagenas", email = "pva@zurich.ibm.com" },
  { name = "Peter Staar", email = "taa@zurich.ibm.com" },
]
requires-python = '>=3.10,<4.0'

# MINIMAL BASE (8 packages)
dependencies = [
  'pydantic>=2.0.0,<3.0.0',
  'docling-core>=2.70.0,<3.0.0',
  'pydantic-settings>=2.3.0,<3.0.0',
  'filetype>=1.2.0,<2.0.0',
  'requests>=2.32.2,<3.0.0',
  'certifi>=2024.7.4',
  'pluggy>=1.0.0,<2.0.0',
  'tqdm>=4.65.0,<5.0.0',
]

[project.urls]
homepage = "https://github.com/docling-project/docling"
repository = "https://github.com/docling-project/docling"
issues = "https://github.com/docling-project/docling/issues"
changelog = "https://github.com/docling-project/docling/blob/main/CHANGELOG.md"

[project.entry-points.docling]
"docling_defaults" = "docling.models.plugins.defaults"

# No CLI scripts in slim - see packages/docling/pyproject.toml for CLI

[project.optional-dependencies]
# Core parsing components (fine-grained)
parse-core = [
  'numpy>=1.24.0,<3.0.0',
  'pillow>=10.0.0,<13.0.0',
]

parse-pdf = [
  'docling-parse>=5.3.2,<6.0.0',
  'pypdfium2>=4.30.0,!=4.30.1,<6.0.0',
]

parse-spatial = [
  'rtree>=1.3.0,<2.0.0',
]

parse = [
  'docling-slim[parse-core,parse-pdf,parse-spatial]',
]

# Model dependencies (fine-grained)
models-core = [
  'scipy>=1.6.0,<2.0.0',
  'rtree>=1.3.0,<2.0.0',
]

models-inference = [
  'torch>=2.2.2,<3.0.0',
  'torchvision>=0,<1',
  'docling-ibm-models>=3.13.0,<4',
  'accelerate>=1.0.0,<2',
  'huggingface_hub>=0.23,<2',
  'defusedxml>=0.7.1,<0.8.0',
]

models = [
  'docling-slim[parse,models-core,models-inference]',
]

# OCR engines (separate extras)
ocr-rapidocr = [
  'rapidocr>=3.3,<4.0.0',
]

ocr-rapidocr-onnx = [
  'rapidocr>=3.3,<4.0.0',
  'onnxruntime>=1.7.0,<2.0.0 ; python_version < "3.14"',
]

ocr-easyocr = [
  'easyocr>=1.7,<2.0',
]

ocr-tesserocr = [
  'tesserocr>=2.7.1,<3.0.0',
  'pandas>=2.1.4,<4.0.0',
]

ocr-mac = [
  'ocrmac>=1.0.0,<2.0.0 ; sys_platform == "darwin"',
]

# Input format support (fine-grained)
format-docx = [
  'python-docx>=1.1.2,<2.0.0',
]

format-pptx = [
  'python-pptx>=1.0.2,<2.0.0',
]

format-xlsx = [
  'openpyxl>=3.1.5,<4.0.0',
]

format-office = [
  'docling-slim[format-docx,format-pptx,format-xlsx]',
]

format-html = [
  'beautifulsoup4>=4.12.3,<5.0.0',
  'lxml>=4.0.0,<7.0.0',
]

format-markdown = [
  'marko>=2.1.2,<3.0.0',
]

format-web = [
  'docling-slim[format-html,format-markdown]',
]

# Advanced features
vlm = [
  'transformers>=4.42.0,<6.0.0,!=5.0.*,!=5.1.*,!=5.2.*,!=5.3.*',
  'accelerate>=1.2.1,<2.0.0',
  'mlx-vlm>=0.3.0,<1.0.0 ; python_version >= "3.10" and sys_platform == "darwin" and platform_machine == "arm64"',
  'qwen-vl-utils>=0.0.11',
]

asr = [
  'mlx-whisper>=0.4.3 ; python_version >= "3.10" and sys_platform == "darwin" and platform_machine == "arm64"',
  'openai-whisper>=20250625',
  'numba>=0.63.0',
]

htmlrender = [
  'playwright>=1.58.0',
]

xbrl = [
  'arelle-release>=2.38.17,<3.0.0',
]

remote-serving = [
  'tritonclient[grpc]>=2.65.0,<3.0.0',
]

onnxruntime = [
  'onnxruntime<1.24 ; python_version < "3.14" and sys_platform == "darwin"',
  'onnxruntime-gpu<1.24 ; python_version < "3.14" and (sys_platform == "linux" or sys_platform == "win32")',
]

latex = [
  'pylatexenc>=2.10,<3.0',
]

# CLI REMOVED FROM SLIM - moved to full docling package
# cli = [
#   'typer>=0.12.5,<0.22.0',
# ]

chunking = [
  'docling-core[chunking]>=2.70.0,<3.0.0',
]

polyfactory = [
  'polyfactory>=2.22.2',
]

# Convenience extras (CLI removed from standard)
standard = [
  'docling-slim[parse,models,ocr-rapidocr,format-office,format-web,latex,chunking,polyfactory]',
]

all = [
  'docling-slim[standard,vlm,asr,htmlrender,xbrl,remote-serving,onnxruntime,ocr-easyocr,ocr-tesserocr,ocr-mac]',
]

[dependency-groups]
dev = [
  "pre-commit~=3.7",
  "mypy~=1.10",
  "types-setuptools~=70.3",
  "pandas-stubs~=2.1",
  "types-openpyxl~=3.1",
  "types-requests~=2.31",
  "boto3-stubs~=1.37",
  "types-urllib3~=1.26",
  "types-tqdm~=4.67",
  "coverage~=7.6",
  "pytest~=8.3",
  "pytest-cov>=6.1.1",
  "pytest-dependency~=0.6",
  "pytest-durations~=1.6.1",
  "pytest-xdist~=3.3",
  "ipykernel~=6.29",
  "ipywidgets~=8.1",
  "nbqa~=1.9",
  "python-semantic-release~=7.32",
  "types-defusedxml>=0.7.0.20250822,<0.8.0",
]

docs = [
  "mkdocs-material~=9.5",
  "mkdocs-jupyter>=0.25,<0.26",
  "mkdocs-click~=0.8",
  "mkdocs-redirects~=1.2",
  "mkdocstrings[python]~=0.27",
  "griffe-pydantic~=1.1",
]

examples = [
  "datasets~=2.21",
  "python-dotenv~=1.0",
  "langchain-huggingface>=0.0.3",
  "langchain-milvus~=0.1",
  "langchain-text-splitters>=0.2",
  "modelscope>=1.29.0",
  'gliner>=0.2.21 ; python_version < "3.14"',
]

constraints = [
  'numba>=0.63.0',
  'langchain-core>=0.3.81',
  'pandas>=2.1.4,<3.0.0 ; python_version < "3.11"',
  'pandas>=2.1.4,<4.0.0 ; python_version >= "3.11"',
]

[tool.hatch.build.targets.wheel]
packages = ["docling"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### packages/docling/pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "docling"
version = "2.85.0"  # DO NOT EDIT, updated automatically
description = "Full-featured SDK and CLI for parsing PDF, DOCX, HTML, and more"
license = "MIT"
keywords = [
  "docling",
  "convert",
  "document",
  "pdf",
  "docx",
  "html",
  "markdown",
  "layout model",
  "segmentation",
  "table structure",
  "table former",
]
classifiers = [
  "Operating System :: MacOS :: MacOS X",
  "Operating System :: POSIX :: Linux",
  "Operating System :: Microsoft :: Windows",
  "Development Status :: 5 - Production/Stable",
  "Intended Audience :: Developers",
  "Intended Audience :: Science/Research",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Programming Language :: Python :: 3.14",
]
readme = "README.md"
authors = [
  { name = "Christoph Auer", email = "cau@zurich.ibm.com" },
  { name = "Michele Dolfi", email = "dol@zurich.ibm.com" },
  { name = "Maxim Lysak", email = "mly@zurich.ibm.com" },
  { name = "Nikos Livathinos", email = "nli@zurich.ibm.com" },
  { name = "Ahmed Nassar", email = "ahn@zurich.ibm.com" },
  { name = "Panos Vagenas", email = "pva@zurich.ibm.com" },
  { name = "Peter Staar", email = "taa@zurich.ibm.com" },
]
requires-python = '>=3.10,<4.0'

# Point to shared source code at repository root
packages = ["docling"]

# DEPENDS ON DOCLING-SLIM WITH CURRENT DEFAULT EXTRAS + CLI
dependencies = [
  'docling-slim[standard]==2.85.0',
  'typer>=0.12.5,<0.22.0',  # CLI dependency
]

[project.urls]
homepage = "https://github.com/docling-project/docling"
repository = "https://github.com/docling-project/docling"
issues = "https://github.com/docling-project/docling/issues"
changelog = "https://github.com/docling-project/docling/blob/main/CHANGELOG.md"

[project.entry-points.docling]
"docling_defaults" = "docling.models.plugins.defaults"

[project.scripts]
docling = "docling.cli.main:app"
docling-tools = "docling.cli.tools:app"

[tool.uv.sources]
# For local development: use workspace member
docling-slim = { workspace = true }

# Re-export slim extras for convenience
[project.optional-dependencies]
easyocr = ['docling-slim[ocr-easyocr]==2.85.0']
tesserocr = ['docling-slim[ocr-tesserocr]==2.85.0']
ocrmac = ['docling-slim[ocr-mac]==2.85.0']
vlm = ['docling-slim[vlm]==2.85.0']
rapidocr = ['docling-slim[ocr-rapidocr-onnx]==2.85.0']
asr = ['docling-slim[asr]==2.85.0']
htmlrender = ['docling-slim[htmlrender]==2.85.0']
remote-serving = ['docling-slim[remote-serving]==2.85.0']
onnxruntime = ['docling-slim[onnxruntime]==2.85.0']

[dependency-groups]
dev = [
    "pre-commit~=3.7",
    "mypy~=1.10",
    "types-setuptools~=70.3",
    "pandas-stubs~=2.1",
    "types-openpyxl~=3.1",
    "types-requests~=2.31",
    "boto3-stubs~=1.37",
    "types-urllib3~=1.26",
    "types-tqdm~=4.67",
    "coverage~=7.6",
    "pytest~=8.3",
    "pytest-cov>=6.1.1",
    "pytest-dependency~=0.6",
    "pytest-durations~=1.6.1",
    "pytest-xdist~=3.3",
    "ipykernel~=6.29",
    "ipywidgets~=8.1",
    "nbqa~=1.9",
    "python-semantic-release~=7.32",
    "types-defusedxml>=0.7.0.20250822,<0.8.0",
]
docs = [
  "mkdocs-material~=9.5",
  "mkdocs-jupyter>=0.25,<0.26",
  "mkdocs-click~=0.8",
  "mkdocs-redirects~=1.2",
  "mkdocstrings[python]~=0.27",
  "griffe-pydantic~=1.1",
]
examples = [
  "datasets~=2.21",
  "python-dotenv~=1.0",
  "langchain-huggingface>=0.0.3",
  "langchain-milvus~=0.1",
  "langchain-text-splitters>=0.2",
  "modelscope>=1.29.0",
  'gliner>=0.2.21 ; python_version < "3.14"',
]
constraints = [
  'numba>=0.63.0',
  'langchain-core>=0.3.81',
  'pandas>=2.1.4,<3.0.0 ; python_version < "3.11"',
  'pandas>=2.1.4,<4.0.0 ; python_version >= "3.11"',
]

[tool.hatch.build.targets.wheel]
packages = ["docling"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Build Process

```bash
# build-packages.sh
#!/bin/bash
set -e

# Build docling-slim from its package directory
echo "Building docling-slim..."
cd packages/docling-slim
uv build --out-dir ../../dist
cd ../..

# Build docling from its package directory
echo "Building docling..."
cd packages/docling
uv build --out-dir ../../dist
cd ../..

echo "Built packages:"
ls -lh dist/
```

## CI/CD Workflow

### Modified .github/workflows/pypi.yml

```yaml
name: "Build and publish packages"

on:
  release:
    types: [published]

env:
  UV_FROZEN: "1"

permissions:
  contents: read

jobs:
  build-and-publish-slim:
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/docling-slim
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: '3.12'
          enable-cache: true
      
      - name: Build docling-slim
        run: |
          cd packages/docling-slim
          uv build --out-dir ../../dist
      
      - name: Publish docling-slim to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist/
          attestations: true

  build-and-publish-full:
    needs: build-and-publish-slim
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/docling
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: '3.12'
          enable-cache: true
      
      - name: Wait for docling-slim availability
        run: |
          VERSION=$(grep '^version = ' packages/docling-slim/pyproject.toml | cut -d'"' -f2)
          echo "Waiting for docling-slim==${VERSION}..."
          for i in {1..50}; do
            if pip index versions docling-slim | grep -q "${VERSION}"; then
              echo "Available!"
              break
            fi
            sleep 10
          done
      
      - name: Build docling
        run: |
          cd packages/docling
          uv build --out-dir ../../dist
      
      - name: Publish docling to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist/
          attestations: true
```

### Modified .github/scripts/release.sh

```bash
#!/bin/bash
set -e
set -x

if [ -z "${TARGET_VERSION}" ]; then
    >&2 echo "No TARGET_VERSION specified"
    exit 1
fi

# Update version in both package pyproject files
echo "Updating versions to ${TARGET_VERSION}..."
uvx --from=toml-cli toml set --toml-path=packages/docling-slim/pyproject.toml project.version "${TARGET_VERSION}"
uvx --from=toml-cli toml set --toml-path=packages/docling/pyproject.toml project.version "${TARGET_VERSION}"

# Update workspace root version (optional, for consistency)
uvx --from=toml-cli toml set --toml-path=pyproject.toml project.version "${TARGET_VERSION}"

# Update docling's dependency on docling-slim
DEPS="docling-slim[standard]==${TARGET_VERSION}"
uvx --from=toml-cli toml set --toml-path=packages/docling/pyproject.toml project.dependencies.0 "${DEPS}"

# Update all optional dependencies using embedded Python script
python3 << 'EOF'
import sys
import os
import re

# Get version from environment
TARGET_VERSION = os.environ.get('TARGET_VERSION')
if not TARGET_VERSION:
    print("ERROR: TARGET_VERSION not set", file=sys.stderr)
    sys.exit(1)

try:
    import tomllib
except ImportError:
    # Fallback for Python < 3.11
    import tomli as tomllib

import tomli_w

# Read packages/docling/pyproject.toml
with open('packages/docling/pyproject.toml', 'rb') as f:
    data = tomllib.load(f)

# Update all optional dependencies that reference docling-slim
optional_deps = data['project'].get('optional-dependencies', {})
updated_count = 0

for extra_name, deps in optional_deps.items():
    if isinstance(deps, list) and len(deps) > 0:
        # Check if this dependency references docling-slim
        dep = deps[0]
        if isinstance(dep, str) and dep.startswith('docling-slim['):
            # Extract the slim extra name and update version
            # Pattern: docling-slim[extra-name]==version
            match = re.match(r'docling-slim\[([^\]]+)\]==.*', dep)
            if match:
                slim_extra = match.group(1)
                deps[0] = f"docling-slim[{slim_extra}]=={TARGET_VERSION}"
                updated_count += 1

# Write back
with open('packages/docling/pyproject.toml', 'wb') as f:
    tomli_w.dump(data, f)

print(f"✓ Updated {updated_count} optional dependencies to version {TARGET_VERSION}")
EOF

# Lock packages
UV_FROZEN=0 uv lock --upgrade-package docling-slim

# Collect release notes and update changelog
# ... (same as before)

# Commit and push
git add packages/*/pyproject.toml pyproject.toml uv.lock CHANGELOG.md
git commit -m "chore: bump version to ${TARGET_VERSION} [skip ci]"
git push origin main

# Create release
gh release create "v${TARGET_VERSION}" -F "${REL_NOTES}"
```

## Local Development Workflow

### **How uv Workspace Works**

The workspace is defined in the root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["packages/docling-slim", "packages/docling"]
```

This tells uv:
1. Both `packages/docling-slim` and `packages/docling` are workspace members
2. When you run `uv sync`, both packages are installed in editable mode
3. Dependencies between workspace members are resolved locally (no PyPI needed)
4. Changes to source code or package metadata are immediately reflected
```

This tells uv:
1. Both `packages/docling-slim` and `packages/docling` are workspace members
2. When you run `uv sync`, both packages are installed in editable mode
3. Dependencies between workspace members are resolved locally (no PyPI needed)
4. Changes to source code or package metadata are immediately reflected

### **Standard Development Workflow**

```bash
# 1. Clone repository
git clone https://github.com/docling-project/docling.git
cd docling

# 2. Sync dependencies - automatically uses local workspace members
uv sync

# 3. Make changes to code
# Edit docling/*.py files (shared source code)
# Or edit packages/docling-slim/pyproject.toml
# Or edit packages/docling/pyproject.toml

# 4. Changes are immediately reflected - no rebuild needed
uv run pytest
uv run docling <your-args>

# 5. If you modify pyproject.toml files, re-sync
uv sync
```

**How it works:**
- `uv sync` reads the workspace configuration from root `pyproject.toml`
- Both `docling-slim` and `docling` are installed as editable packages
- `docling` depends on `docling-slim`, but uv resolves it to the local workspace member
- All changes to source code in `docling/` are immediately visible to both packages
- Changes to package metadata require `uv sync` to update the environment

### **Testing Package Builds**

When you need to test the actual built packages:

```bash
# Build both packages
cd packages/docling-slim && uv build --out-dir ../../dist && cd ../..
cd packages/docling && uv build --out-dir ../../dist && cd ../..

# Or use the build script
./build-packages.sh

# Test installations in a clean environment
uv pip install dist/docling_slim-*.whl
uv pip install dist/docling-*.whl
```

### **Benefits of Workspace Approach**

✅ **Seamless development** - `uv sync` just works with proper workspace members
✅ **No circular dependencies** - workspace resolution handles local dependencies
✅ **Immediate changes** - source changes reflected instantly, config changes need `uv sync`
✅ **Single command** - no manual package building during development
✅ **Standard uv pattern** - follows documented uv workspace behavior
✅ **Shared source code** - both packages reference same `docling/` directory

## Import Audit Requirements (NEW - CRITICAL)

Before releasing `docling-slim`, a comprehensive import audit is required to ensure the base package can be imported without heavy optional dependencies.

### **Audit Checklist**

- [ ] **Base import test**: Verify `import docling` works with only base dependencies
- [ ] **Module-level imports**: Check that top-level `__init__.py` files don't import optional modules
- [ ] **Lazy imports**: Ensure optional backends/models use runtime imports, not module-level
- [ ] **Guard patterns**: Verify all optional functionality has proper try/except or conditional imports
- [ ] **CLI isolation**: Confirm CLI code doesn't get imported when using library-only

### **Test Strategy**

```bash
# Create clean environment with only base dependencies
uv venv test-env
source test-env/bin/activate
uv pip install pydantic docling-core pydantic-settings filetype requests certifi pluggy tqdm

# Test basic imports
python -c "import docling"
python -c "from docling.datamodel import DoclingDocument"
python -c "from docling import DocumentConverter"  # Should work or fail gracefully

# Test that optional imports are guarded
python -c "import docling.models"  # Should not fail if models are optional
python -c "import docling.backend"  # Should not fail if backends are optional
```

### **Common Issues to Fix**

1. **Module-level imports of optional deps**
   ```python
   # BAD - fails if torch not installed
   import torch
   from docling_ibm_models import LayoutModel
   
   # GOOD - lazy import
   def get_layout_model():
       import torch
       from docling_ibm_models import LayoutModel
       return LayoutModel()
   ```

2. **Unguarded optional imports**
   ```python
   # BAD
   from docling.models.layout import LayoutModel
   
   # GOOD
   try:
       from docling.models.layout import LayoutModel
   except ImportError:
       LayoutModel = None
   ```

3. **CLI imports in library code**
   ```python
   # BAD - typer imported at module level
   import typer
   
   # GOOD - only import in CLI entry point
   def main():
       import typer
       app = typer.Typer()
   ```

### **Validation Criteria**

✅ Base `docling-slim` installs successfully with only 8 dependencies
✅ `import docling` succeeds without optional dependencies
✅ Core data models are accessible
✅ Optional features fail gracefully with clear error messages
✅ No import-time failures due to missing optional dependencies

## Migration Guide

### For Existing Users

**No changes needed!**

```bash
pip install docling  # Still works, installs same dependencies as before
```

### For New Users Wanting Minimal Installation

```bash
# Minimal base - just data models (~50MB)
pip install docling-slim

# Add PDF parsing with docling-parse backend (~200MB)
pip install docling-slim[parse]

# OR choose specific backend
pip install docling-slim[backend-pypdfium2]  # Basic PDF parsing
pip install docling-slim[backend-docling-parse]  # Advanced PDF parsing

# Add specific Office format with PDF support
pip install docling-slim[backend-docling-parse,format-docx]

# Add models for local inference (~2.5GB)
pip install docling-slim[backend-docling-parse,models]

# Standard installation (same as docling)
pip install docling-slim[standard]
# OR
pip install docling
```

### Feature Matrix

| Feature | docling-slim base | Extra needed | docling |
|---------|------------------|--------------|---------|
| Core data models | ✅ | - | ✅ |
| PDF parsing (basic) | ❌ | `[backend-pypdfium2]` | ✅ |
| PDF parsing (advanced) | ❌ | `[backend-docling-parse]` or `[parse]` | ✅ |
| Spatial indexing | ❌ | `[parse-spatial]` | ✅ |
| Local model inference | ❌ | `[models]` | ✅ |
| RapidOCR | ❌ | `[ocr-rapidocr]` | ✅ |
| EasyOCR | ❌ | `[ocr-easyocr]` | ❌ (extra) |
| Word docs | ❌ | `[format-docx]` or `[format-office]` | ✅ |
| Excel docs | ❌ | `[format-xlsx]` or `[format-office]` | ✅ |
| PowerPoint | ❌ | `[format-pptx]` or `[format-office]` | ✅ |
| HTML | ❌ | `[format-html]` or `[format-web]` | ✅ |
| Markdown | ❌ | `[format-markdown]` or `[format-web]` | ✅ |
| LaTeX | ❌ | `[latex]` | ✅ |
| XBRL (financial) | ❌ | `[xbrl]` | ❌ (extra) |
| **CLI tools** | ❌ | **Install `docling` package** | ✅ |
| Information extraction | ❌ | `[polyfactory]` | ✅ |
| VLM support | ❌ | `[vlm]` | ❌ (extra) |

**Note:** CLI tools (docling, docling-tools) are only available in the full `docling` package, not in `docling-slim`.

## Size Comparison (Estimated)

| Package | Dependencies | Disk Size | Use Case |
|---------|-------------|-----------|----------|
| **docling-slim** (base) | 8 | ~50MB | Data models only |
| **docling-slim[backend-pypdfium2]** | 11 | ~150MB | + Basic PDF parsing |
| **docling-slim[backend-docling-parse]** | 12 | ~180MB | + Advanced PDF parsing |
| **docling-slim[parse]** | 13 | ~200MB | + Complete parsing (docling-parse + spatial) |
| **docling-slim[parse,format-docx]** | 14 | ~220MB | + Word documents |
| **docling-slim[models]** | 19 | ~2.5GB | + Local ML inference |
| **docling-slim[standard]** | 27 | ~2.8GB | Current default |
| **docling** | 27 | ~2.8GB | Full featured (unchanged) |

## Implementation Checklist

### Phase 1: Repository Structure
- [ ] Create `packages/docling-slim/` directory
- [ ] Create `packages/docling/` directory
- [ ] Create `packages/docling-slim/pyproject.toml` for docling-slim package
- [ ] Create `packages/docling/pyproject.toml` for docling package
- [ ] Update root `pyproject.toml` with workspace configuration
- [ ] Verify source code remains in `docling/` at repository root

### Phase 2: Package Configuration
- [ ] Configure docling-slim with base dependencies only (no CLI)
- [ ] Configure docling-slim extras (all library features)
- [ ] Configure docling to depend on docling-slim[standard]
- [ ] Add CLI dependencies (typer) to docling package
- [ ] Move CLI scripts to docling package only
- [ ] Add `[tool.hatch.build.targets.wheel]` to both packages pointing to shared source

### Phase 3: Build & Release Infrastructure
- [ ] Create `build-packages.sh` script for building from package directories
- [ ] Update `.github/workflows/pypi.yml` to build from package directories
- [ ] Update `.github/scripts/release.sh` to update both package pyproject files
- [ ] Update version sync logic for workspace structure
- [ ] Test build process locally

### Phase 4: Import Audit (CRITICAL)
- [ ] Audit all module-level imports for optional dependencies
- [ ] Add lazy imports for optional backends
- [ ] Add try/except guards for optional features
- [ ] Ensure CLI code is not imported in library usage
- [ ] Test base slim import with minimal dependencies
- [ ] Document any breaking changes or limitations

### Phase 5: Testing & Documentation
- [ ] Update `.github/workflows/ci.yml` to test both packages
- [ ] Test workspace development workflow (`uv sync`)
- [ ] Test building both packages
- [ ] Update README.md with new installation options
- [ ] Create migration guide documentation
- [ ] Document import audit results
- [ ] Test on test.pypi.org

### Phase 6: Release
- [ ] Final review of all changes
- [ ] Verify backward compatibility
- [ ] Test on test.pypi.org
- [ ] Official release to PyPI

## Key Decisions

- **Repository Structure**: Real workspace members in `packages/` subdirectories
- **Source Code Location**: Remains in `docling/` at repository root (shared by both packages)
- **Workspace Configuration**: Standard uv workspace in root `pyproject.toml`
- **Version Pinning**: Exact pinning (docling depends on docling-slim==X.Y.Z)
- **CLI Location**: In docling package only (not in slim)
- **docling Dependencies**: Same as current default (docling-slim[standard] + typer)
- **Extras Design**: Fine-grained, composable extras
- **Build Process**: Build from package directories

## Benefits

- **95% smaller** minimal installation (50MB vs 2.8GB)
- **Zero breaking changes** for existing users
- **Maximum flexibility** with fine-grained extras
- **Composable installation** - users pick exactly what they need
- **Source code stays in place** - only metadata moves
- **Standard uv workspace** - follows documented patterns
- **Proper Python packaging** - no conditional console scripts

## Next Steps

1. Begin implementation following this structure
2. Complete import audit before release
3. Test thoroughly with workspace development workflow
4. Release to PyPI