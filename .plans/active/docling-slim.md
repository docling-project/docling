# Docling Slim Refactoring Plan - UPDATED FOR v2.85.0

## Project Context
- **Current Package**: `docling` (version 2.85.0)
- **Goal**: Split into `docling-slim` (minimal dependencies) and `docling` (current default dependencies)
- **Repository**: https://github.com/docling-project/docling

## Key User Requirements (CONFIRMED)

1. **No source code movement** - This is purely a packaging refactoring
2. **pyproject.toml REMAINS docling** - Keep current file as-is (with modifications)
3. **Create pyproject-slim.toml** - New file for docling-slim package
4. **docling depends on docling-slim** with SAME set of dependencies as current docling (not [all])
5. **Exact version pinning** - docling depends on exact version of docling-slim
6. **CLI in docling-slim[cli]** extra
7. **Fine-grained extras** - Split parse and other components into granular options
8. **CI/CD pushes both packages**
9. **Local development** uses local docling-slim changes

## Repository Structure (FINAL - No Code Movement)

```
docling/  (current repository, no restructuring)
├── pyproject.toml              # REMAINS docling (modified to depend on docling-slim)
├── pyproject-slim.toml         # NEW: docling-slim package definition
├── pyproject.workspace.toml    # NEW: uv workspace configuration for local dev
├── docling/                    # Source code (UNCHANGED)
├── tests/                      # Tests (UNCHANGED)
├── docs/                       # Docs (UNCHANGED)
└── .github/
    └── workflows/
        ├── ci.yml              # Modified: test both packages
        ├── cd.yml              # Modified: release both
        └── pypi.yml            # Modified: publish both (slim first, then full)
```

## Dependency Categorization (FINAL - v2.85.0)

### **docling-slim BASE (8 packages)** - ~50MB, Data Models Only

```python
dependencies = [
    'pydantic>=2.0.0,<3.0.0',
    'docling-core>=2.70.0,<3.0.0',  # WITHOUT chunking
    'pydantic-settings>=2.3.0,<3.0.0',
    'filetype>=1.2.0,<2.0.0',
    'requests>=2.32.2,<3.0.0',
    'certifi>=2024.7.4',
    'pluggy>=1.0.0,<2.0.0',
    'tqdm>=4.65.0,<5.0.0',
]
```

**What this provides:**
- Core data models (DoclingDocument, ConversionResult)
- Document format definitions
- Basic I/O utilities
- **NO document processing** - just data structures

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

**`[standard]` - Standard PDF processing (matches current docling default)**
```python
'docling-slim[parse,models,ocr-rapidocr,format-office,format-web,format-latex,cli,chunking,extraction]',
```

**`[all]` - Everything**
```python
'docling-slim[parse,models,ocr-rapidocr,ocr-rapidocr-onnx,ocr-easyocr,ocr-tesserocr,ocr-mac,format-office,format-web,format-latex,format-xbrl,vlm,asr,cli,chunking,extraction,htmlrender,remote-serving,onnxruntime]',
```

## docling Package Dependencies (FINAL)

The `docling` package should have the SAME dependencies as current docling has now:

```python
dependencies = [
    'docling-slim[standard]==2.85.0',
]
```

This matches the current default installation of docling.

## Detailed pyproject.toml Structures (FINAL)

### pyproject-slim.toml (NEW - docling-slim package)

```toml
[project]
name = "docling-slim"
version = "2.85.0"  # DO NOT EDIT, updated automatically
description = "Lightweight SDK for parsing documents (minimal dependencies, opt-in extras)"
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

[project.scripts]
docling = "docling.cli.main:app"
docling-tools = "docling.cli.tools:app"

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

cli = [
  'typer>=0.12.5,<0.22.0',
]

chunking = [
  'docling-core[chunking]>=2.70.0,<3.0.0',
]

polyfactory = [
  'polyfactory>=2.22.2',
]

# Convenience extras
standard = [
  'docling-slim[parse,models,ocr-rapidocr,format-office,format-web,latex,cli,chunking,polyfactory]',
]

all = [
  'docling-slim[parse,models,ocr-rapidocr,ocr-rapidocr-onnx,ocr-easyocr,ocr-tesserocr,ocr-mac,format-office,format-web,vlm,asr,latex,cli,chunking,polyfactory,htmlrender,xbrl,remote-serving,onnxruntime]',
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

[tool.uv]
package = true
default-groups = "all"

[tool.setuptools.packages.find]
include = ["docling*"]

# Copy all tool configurations from pyproject.toml (ruff, mypy, etc.)
```

### pyproject.toml (REMAINS docling - modified)

```toml
[project]
name = "docling"
version = "2.85.0"  # DO NOT EDIT, updated automatically
description = "SDK and CLI for parsing PDF, DOCX, HTML, and more, to a unified document representation"
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

# DEPENDS ON DOCLING-SLIM WITH CURRENT DEFAULT EXTRAS
dependencies = [
  'docling-slim[standard]==2.85.0',
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

# Re-export extras for convenience
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

[tool.uv]
package = true
default-groups = "all"

[tool.uv.sources]
docling-slim = { workspace = true }

# Keep all existing tool configurations (ruff, mypy, etc.)
```

## Build Process (FINAL)

```bash
# build-packages.sh
#!/bin/bash
set -e

# Build docling-slim (uses pyproject-slim.toml)
echo "Building docling-slim..."
uv build --config-file pyproject-slim.toml --out-dir dist-slim

# Build docling (uses pyproject.toml - current)
echo "Building docling..."
uv build --out-dir dist-full
```

## CI/CD Workflow (FINAL)

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
          uv build --config-file pyproject-slim.toml --out-dir dist-slim
      
      - name: Publish docling-slim to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist-slim/
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
          VERSION=$(grep '^version = ' pyproject-slim.toml | cut -d'"' -f2)
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
          uv build --out-dir dist-full
      
      - name: Publish docling to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist-full/
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

# Update version in both pyproject files
echo "Updating versions to ${TARGET_VERSION}..."
uvx --from=toml-cli toml set --toml-path=pyproject-slim.toml project.version "${TARGET_VERSION}"
uvx --from=toml-cli toml set --toml-path=pyproject.toml project.version "${TARGET_VERSION}"

# Update docling's dependency on docling-slim
DEPS="docling-slim[standard]==${TARGET_VERSION}"
uvx --from=toml-cli toml set --toml-path=pyproject.toml project.dependencies.0 "${DEPS}"

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

# Read pyproject.toml
with open('pyproject.toml', 'rb') as f:
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
with open('pyproject.toml', 'wb') as f:
    tomli_w.dump(data, f)

print(f"✓ Updated {updated_count} optional dependencies to version {TARGET_VERSION}")
EOF

# Lock packages
UV_FROZEN=0 uv lock --upgrade-package docling-slim

# Collect release notes and update changelog
# ... (same as before)

# Commit and push
git add pyproject-slim.toml pyproject.toml uv.lock CHANGELOG.md
git commit -m "chore: bump version to ${TARGET_VERSION} [skip ci]"
git push origin main

# Create release
gh release create "v${TARGET_VERSION}" -F "${REL_NOTES}"
```

**Key Improvements:**
- **Zero maintenance**: Automatically discovers and updates all `docling-slim[*]` dependencies
- **No hardcoded extras**: Works with any optional dependencies that reference docling-slim
- **Atomic operation**: All updates happen in one Python execution
- **Type-safe**: Uses proper TOML libraries (`tomllib`/`tomli` for reading, `tomli_w` for writing)
- **Clear feedback**: Prints confirmation of how many extras were updated
- **Python 3.10+ compatible**: Falls back to `tomli` for Python < 3.11
- **Future-proof**: Adding/removing extras in pyproject.toml automatically works

**Dependencies**: Requires `tomli` and `tomli-w` packages (can be installed via `pip install tomli tomli-w` or added to CI environment)

## Local Development Workflow (FINAL)

### **Workspace Configuration**

Create `pyproject.workspace.toml` in the repository root:

```toml
[tool.uv.workspace]
members = [".", "slim"]

[tool.uv.workspace.package-sources]
docling-slim = { path = ".", config = "pyproject-slim.toml" }
```

This tells uv that:
- The workspace has two members: the main package (`.`) and slim variant (`slim` virtual member)
- `docling-slim` should be resolved from the local source using `pyproject-slim.toml`

### **Standard Development Workflow (Recommended)**

```bash
# 1. Clone repository
git clone https://github.com/docling-project/docling.git
cd docling

# 2. Sync dependencies - automatically uses local docling-slim
uv sync --all-extras

# 3. Make changes to code
# Edit docling/*.py files or pyproject-slim.toml

# 4. Changes are immediately reflected - no rebuild needed
uv run pytest
uv run docling <your-args>

# 5. Update dependencies if pyproject-slim.toml changes
uv sync
```

**How it works:**
- `uv sync` reads `pyproject.workspace.toml` and recognizes the workspace
- When resolving `docling-slim` dependency, it uses the local source with `pyproject-slim.toml`
- Both packages are installed in editable mode
- All changes to source code or `pyproject-slim.toml` are immediately visible
- No circular dependency because workspace resolution handles it correctly

### **Testing Package Builds**

When you need to test the actual built packages:

```bash
# Build both packages
uv build --config-file pyproject-slim.toml --out-dir dist-slim
uv build --out-dir dist-full

# Test installations in a clean environment
uv pip install dist-slim/docling_slim-*.whl
uv pip install dist-full/docling-*.whl
```

### **Benefits of Workspace Approach**

✅ **Seamless development** - `uv sync` just works
✅ **No circular dependencies** - workspace resolution handles it
✅ **Immediate changes** - both source and config changes reflected instantly
✅ **Single command** - no manual package building during development
✅ **CI/CD compatible** - can disable workspace for production builds

## Migration Guide (FINAL)

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

### Feature Matrix (FINAL)

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
| LaTeX | ❌ | `[format-latex]` | ✅ |
| XBRL (financial) | ❌ | `[format-xbrl]` | ❌ (extra) |
| CLI tools | ❌ | `[cli]` | ✅ |
| Information extraction | ❌ | `[extraction]` | ✅ |
| VLM support | ❌ | `[vlm]` | ❌ (extra) |

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

- [ ] Create `pyproject-slim.toml` for docling-slim package
- [ ] Modify `pyproject.toml` to depend on docling-slim
- [ ] Update dependency lists according to new extras structure
- [ ] Create build script for both packages
- [ ] Update `.github/workflows/pypi.yml`
- [ ] Update `.github/scripts/release.sh`
- [ ] Update `.github/workflows/ci.yml` to test both packages
- [ ] Update README.md with new installation options
- [ ] Create migration guide documentation
- [ ] Test local builds
- [ ] Test on test.pypi.org
- [ ] Official release

## Key Decisions Confirmed

✅ **Repository Structure**: No code movement, just packaging changes
✅ **pyproject.toml**: REMAINS docling (modified to depend on docling-slim)
✅ **pyproject-slim.toml**: NEW file for docling-slim
✅ **Version Pinning**: Exact pinning (docling depends on docling-slim==X.Y.Z)
✅ **CLI Location**: In docling-slim[cli] extra
✅ **docling Dependencies**: Same as current default (not [all])
✅ **Extras Design**: Fine-grained, composable extras
✅ **PDF Backends**: Separate extras for pypdfium2 vs docling-parse backends
✅ **Parse Component**: backend-pypdfium2, backend-docling-parse, parse-spatial, parse (convenience)
✅ **Format Support**: Split into individual format extras (docx, pptx, xlsx, html, markdown, latex, xbrl)
✅ **Extraction**: polyfactory moved to [extraction] extra for information extraction functionality

## Benefits

- **95% smaller** minimal installation (50MB vs 2.8GB)
- **Zero breaking changes** for existing users
- **Maximum flexibility** with fine-grained extras
- **Composable installation** - users pick exactly what they need
- **No code movement** - simpler implementation
- **Minimal disruption** to current development workflow

## Next Steps

1. ✅ Review this final plan
2. ⚠️ Begin implementation
3. ⚠️ Test thoroughly before release