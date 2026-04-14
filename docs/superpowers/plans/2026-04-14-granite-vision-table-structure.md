# Granite Vision Table Structure Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `GraniteVisionTableStructureModel` — a new table structure backend that uses `ibm-granite/granite-4.0-3b-vision` with the `<tables_otsl>` prompt to extract table structure and cell text from document images.

**Architecture:** A standalone class inheriting `BaseTableStructureModel`, following the same model-loading pattern as `ChartExtractionModelGraniteVisionV4`. Each page's table clusters are cropped from the page image, batched, and run through the VLM, whose OTSL text output is parsed into `TableCell` objects.

**Tech Stack:** Python, PyTorch, HuggingFace `transformers` (`AutoProcessor`, `AutoModelForImageTextToText`), `docling_core`, PIL

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `docling/models/stages/table_structure/table_structure_model_granite_vision.py` | New model class + OTSL parser |
| Modify | `docling/datamodel/pipeline_options.py:144` | Add `GraniteVisionTableStructureOptions` after `TableStructureV2Options` |
| Modify | `docling/models/plugins/defaults.py:61-74` | Register `GraniteVisionTableStructureModel` in `table_structure_engines()` |
| Create | `tests/test_table_structure_granite_vision.py` | Unit tests for OTSL parser and model integration |

---

## Task 1: Add `GraniteVisionTableStructureOptions` to pipeline_options.py

**Files:**
- Modify: `docling/datamodel/pipeline_options.py:144-154`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_table_structure_granite_vision.py
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions


def test_options_kind():
    opts = GraniteVisionTableStructureOptions()
    assert opts.kind == "granite_vision_table"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/elisch/Documents/repos/docling
python -m pytest tests/test_table_structure_granite_vision.py::test_options_kind -v
```

Expected: `ImportError` or `AttributeError` — `GraniteVisionTableStructureOptions` does not exist yet.

- [ ] **Step 3: Add the options class to `pipeline_options.py`**

Open `docling/datamodel/pipeline_options.py`. After line 154 (end of `TableStructureV2Options`), insert:

```python
class GraniteVisionTableStructureOptions(BaseTableStructureOptions):
    """Options for the table structure model using Granite Vision (VLM-based)."""

    kind: ClassVar[str] = "granite_vision_table"
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
python -m pytest tests/test_table_structure_granite_vision.py::test_options_kind -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add docling/datamodel/pipeline_options.py tests/test_table_structure_granite_vision.py
git commit -m "feat: add GraniteVisionTableStructureOptions"
```

---

## Task 2: Implement `_parse_otsl_output()`

This is the core parser that converts VLM text output into structured table data. Build and test it in isolation before wiring up the model.

**Files:**
- Create: `docling/models/stages/table_structure/table_structure_model_granite_vision.py`
- Modify: `tests/test_table_structure_granite_vision.py`

- [ ] **Step 1: Write failing tests for the parser**

Add to `tests/test_table_structure_granite_vision.py`:

```python
from docling.models.stages.table_structure.table_structure_model_granite_vision import (
    _parse_otsl_output,
)


def test_parse_simple_table():
    """2x2 table with column headers."""
    text = "<ched>Name</ched><ched>Value</ched><nl><fcel>Foo</fcel><fcel>42</fcel><nl>"
    otsl_seq, cells, num_rows, num_cols = _parse_otsl_output(text)

    assert otsl_seq == ["ched", "ched", "nl", "fcel", "fcel", "nl"]
    assert num_rows == 2
    assert num_cols == 2
    assert len(cells) == 4

    header_cells = [c for c in cells if c.column_header]
    assert len(header_cells) == 2
    assert header_cells[0].text == "Name"
    assert header_cells[1].text == "Value"

    data_cells = [c for c in cells if not c.column_header]
    assert data_cells[0].text == "Foo"
    assert data_cells[0].start_row_offset_idx == 1
    assert data_cells[0].start_col_offset_idx == 0
    assert data_cells[1].text == "42"
    assert data_cells[1].start_col_offset_idx == 1


def test_parse_empty_cell():
    """Empty cell produces empty text, still in grid."""
    text = "<ched>A</ched><ched>B</ched><nl><fcel>x</fcel><ecel></ecel><nl>"
    otsl_seq, cells, num_rows, num_cols = _parse_otsl_output(text)

    assert num_rows == 2
    assert num_cols == 2
    assert len(cells) == 4
    empty = [c for c in cells if c.start_row_offset_idx == 1 and c.start_col_offset_idx == 1]
    assert len(empty) == 1
    assert empty[0].text == ""


def test_parse_colspan():
    """lcel produces colspan=2 on the preceding fcel."""
    text = "<fcel>Merged</fcel><lcel><nl><fcel>A</fcel><fcel>B</fcel><nl>"
    otsl_seq, cells, num_rows, num_cols = _parse_otsl_output(text)

    assert num_cols == 2
    merged = [c for c in cells if c.start_row_offset_idx == 0 and c.start_col_offset_idx == 0]
    assert len(merged) == 1
    assert merged[0].col_span == 2
    assert merged[0].end_col_offset_idx == 2


def test_parse_rowspan():
    """ucel produces rowspan=2 on the preceding fcel above it."""
    text = "<fcel>Tall</fcel><fcel>A</fcel><nl><ucel><fcel>B</fcel><nl>"
    otsl_seq, cells, num_rows, num_cols = _parse_otsl_output(text)

    tall = [c for c in cells if c.start_row_offset_idx == 0 and c.start_col_offset_idx == 0]
    assert len(tall) == 1
    assert tall[0].row_span == 2
    assert tall[0].end_row_offset_idx == 2


def test_parse_row_header():
    """rhed token produces row_header=True."""
    text = "<rhed>Section</rhed><fcel>Data</fcel><nl>"
    _, cells, _, _ = _parse_otsl_output(text)

    rhed_cells = [c for c in cells if c.row_header]
    assert len(rhed_cells) == 1
    assert rhed_cells[0].text == "Section"


def test_parse_no_bbox():
    """All cells must have bbox=None."""
    text = "<ched>X</ched><nl><fcel>Y</fcel><nl>"
    _, cells, _, _ = _parse_otsl_output(text)
    assert all(c.bbox is None for c in cells)


def test_parse_empty_string():
    """Empty or whitespace-only string returns empty table."""
    otsl_seq, cells, num_rows, num_cols = _parse_otsl_output("")
    assert otsl_seq == []
    assert cells == []
    assert num_rows == 0
    assert num_cols == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_table_structure_granite_vision.py -k "parse" -v
```

Expected: `ImportError` — module does not exist yet.

- [ ] **Step 3: Create the new file with `_parse_otsl_output()`**

Create `docling/models/stages/table_structure/table_structure_model_granite_vision.py`:

```python
import logging
import re
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, cast

import torch
from docling_core.types.doc import DocItemLabel, TableCell
from transformers import AutoModelForImageTextToText, AutoProcessor

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
from docling.models.base_table_model import BaseTableStructureModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

# OTSL tokens that represent content-bearing cells (produce a TableCell)
_CONTENT_TOKENS = {"fcel", "ecel", "ched", "rhed", "srow"}
# OTSL tokens that are span-extensions (no separate TableCell, affect span of predecessor)
_SPAN_TOKENS = {"lcel", "ucel", "xcel"}

# Regex to extract (tag_name, inner_text) from VLM OTSL output.
# Matches: <tag>text</tag>  OR  <tag/>  OR  <tag>  (self-closing / bare nl)
_TAG_RE = re.compile(
    r"<(?P<tag>[a-z]+)>(?P<text>.*?)</(?P=tag)>"  # <tag>text</tag>
    r"|<(?P<stag>[a-z]+)\s*/>"                     # <tag/>
    r"|<(?P<btag>[a-z]+)>",                        # <tag> (bare, e.g. <nl>, <lcel>)
    re.DOTALL,
)


def _parse_otsl_output(
    text: str,
) -> tuple[list[str], list[TableCell], int, int]:
    """Parse VLM OTSL text output into structured table data.

    Parameters
    ----------
    text:
        Raw VLM output string, e.g.
        ``"<ched>Name</ched><ched>Val</ched><nl><fcel>Foo</fcel><fcel>42</fcel><nl>"``

    Returns
    -------
    tuple of (otsl_seq, table_cells, num_rows, num_cols)
        otsl_seq: list of bare tag names, e.g. ["ched", "ched", "nl", "fcel", "fcel", "nl"]
        table_cells: list of TableCell (bbox always None)
        num_rows: int
        num_cols: int
    """
    if not text or not text.strip():
        return [], [], 0, 0

    # Extract (tag, inner_text) pairs
    token_pairs: list[tuple[str, str]] = []
    for m in _TAG_RE.finditer(text):
        if m.group("tag"):
            token_pairs.append((m.group("tag"), m.group("text") or ""))
        elif m.group("stag"):
            token_pairs.append((m.group("stag"), ""))
        elif m.group("btag"):
            token_pairs.append((m.group("btag"), ""))

    if not token_pairs:
        return [], [], 0, 0

    otsl_seq = [tag for tag, _ in token_pairs]

    # Split into rows on "nl" tokens
    rows: list[list[tuple[str, str]]] = [
        list(group)
        for k, group in groupby(token_pairs, lambda x: x[0] == "nl")
        if not k
    ]

    if not rows:
        return otsl_seq, [], 0, 0

    num_rows = len(rows)
    num_cols = max(len(row) for row in rows)

    # Pad rows to equal width
    grid: list[list[tuple[str, str]]] = [
        row + [("", "")] * (num_cols - len(row)) for row in rows
    ]

    table_cells: list[TableCell] = []
    for row_idx, row in enumerate(grid):
        for col_idx, (tag, inner_text) in enumerate(row):
            if tag not in _CONTENT_TOKENS:
                continue

            # Detect colspan: count consecutive lcel / xcel to the right
            colspan = 1
            for c in range(col_idx + 1, num_cols):
                if grid[row_idx][c][0] in ("lcel", "xcel"):
                    colspan += 1
                else:
                    break

            # Detect rowspan: count consecutive ucel / xcel below
            rowspan = 1
            for r in range(row_idx + 1, num_rows):
                if grid[r][col_idx][0] in ("ucel", "xcel"):
                    rowspan += 1
                else:
                    break

            cell = TableCell(
                text=inner_text,
                bbox=None,
                row_span=rowspan,
                col_span=colspan,
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + rowspan,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + colspan,
                column_header=(tag == "ched"),
                row_header=(tag == "rhed"),
                row_section=(tag == "srow"),
            )
            table_cells.append(cell)

    return otsl_seq, table_cells, num_rows, num_cols
```

- [ ] **Step 4: Run parser tests**

```bash
python -m pytest tests/test_table_structure_granite_vision.py -k "parse" -v
```

Expected: all 7 parser tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add docling/models/stages/table_structure/table_structure_model_granite_vision.py \
        tests/test_table_structure_granite_vision.py
git commit -m "feat: implement _parse_otsl_output for Granite Vision table model"
```

---

## Task 3: Implement `GraniteVisionTableStructureModel`

**Files:**
- Modify: `docling/models/stages/table_structure/table_structure_model_granite_vision.py`
- Modify: `tests/test_table_structure_granite_vision.py`

- [ ] **Step 1: Write failing test for `get_options_type()`**

Add to `tests/test_table_structure_granite_vision.py`:

```python
from docling.models.stages.table_structure.table_structure_model_granite_vision import (
    GraniteVisionTableStructureModel,
    _parse_otsl_output,
)
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
from docling.datamodel.accelerator_options import AcceleratorOptions


def test_get_options_type():
    assert GraniteVisionTableStructureModel.get_options_type() is GraniteVisionTableStructureOptions


def test_model_disabled_skips_pages():
    """When enabled=False, predict_tables returns empty predictions without loading the model."""
    from unittest.mock import MagicMock
    model = GraniteVisionTableStructureModel(
        enabled=False,
        artifacts_path=None,
        options=GraniteVisionTableStructureOptions(),
        accelerator_options=AcceleratorOptions(),
    )
    page = MagicMock()
    page._backend.is_valid.return_value = True
    page.predictions.layout.clusters = []
    results = model.predict_tables(MagicMock(), [page])
    assert len(results) == 1
    assert results[0].table_map == {}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_table_structure_granite_vision.py -k "options_type or disabled" -v
```

Expected: `ImportError` — `GraniteVisionTableStructureModel` not yet defined.

- [ ] **Step 3: Add `GraniteVisionTableStructureModel` class to the module**

Append to `docling/models/stages/table_structure/table_structure_model_granite_vision.py`:

```python

class GraniteVisionTableStructureModel(BaseTableStructureModel):
    """Table structure model using ibm-granite/granite-4.0-3b-vision with <tables_otsl>."""

    _model_repo_id: ClassVar[str] = "ibm-granite/granite-4.0-3b-vision"
    _model_repo_folder: ClassVar[str] = "ibm-granite--granite-4.0-3b-vision"
    _model_repo_revision: ClassVar[str] = "f0d034897bae1cd438c961c8c170a3a3089ebf01"

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: GraniteVisionTableStructureOptions,
        accelerator_options: AcceleratorOptions,
        enable_remote_services: Literal[False] = False,
    ):
        self.enabled = enabled
        self.options = options

        if self.enabled:
            self.device = decide_device(
                accelerator_options.device,
                supported_devices=[AcceleratorDevice.CPU, AcceleratorDevice.CUDA],
            )

            if artifacts_path is None:
                artifacts_path = self.download_models()
            elif (artifacts_path / self._model_repo_folder).exists():
                artifacts_path = artifacts_path / self._model_repo_folder
            else:
                _log.warning(
                    f"Model artifacts not found at {artifacts_path / self._model_repo_folder},"
                    " they will be downloaded."
                )

            self._load_model(artifacts_path)

    @classmethod
    def get_options_type(cls) -> type[GraniteVisionTableStructureOptions]:
        return GraniteVisionTableStructureOptions

    @classmethod
    def download_models(
        cls,
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        return download_hf_model(
            repo_id=cls._model_repo_id,
            revision=cls._model_repo_revision,
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    def _load_model(self, artifacts_path: Path) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*torch_dtype.*deprecated.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*incorrect regex pattern.*",
                category=UserWarning,
            )
            self._processor = AutoProcessor.from_pretrained(
                artifacts_path,
                trust_remote_code=True,
            )
            self._model_max_length = self._processor.tokenizer.model_max_length
            self._model = AutoModelForImageTextToText.from_pretrained(
                artifacts_path,
                device_map=self.device,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        cast(Any, self._model).merge_lora_adapters()
        self._model.eval()

    def predict_tables(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[TableStructurePrediction]:
        predictions: list[TableStructurePrediction] = []

        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                existing = page.predictions.tablestructure or TableStructurePrediction()
                page.predictions.tablestructure = existing
                predictions.append(existing)
                continue

            with TimeRecorder(conv_res, "table_structure"):
                assert page.predictions.layout is not None
                assert page.size is not None

                table_prediction = TableStructurePrediction()
                page.predictions.tablestructure = table_prediction

                clusters = [
                    c
                    for c in page.predictions.layout.clusters
                    if c.label in (DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX)
                ]

                if not clusters or not self.enabled:
                    predictions.append(table_prediction)
                    continue

                # Crop one image per table cluster
                from docling_core.types.doc import BoundingBox

                crop_images = []
                for cluster in clusters:
                    crop = page.get_image(scale=1.0, cropbox=cluster.bbox)
                    if crop is None:
                        crop_images.append(None)
                    else:
                        crop_images.append(crop)

                # Build batched conversations (skip clusters whose crop failed)
                valid_pairs = [
                    (cluster, img)
                    for cluster, img in zip(clusters, crop_images)
                    if img is not None
                ]

                if not valid_pairs:
                    predictions.append(table_prediction)
                    continue

                valid_clusters, valid_images = zip(*valid_pairs)

                conversations = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "<tables_otsl>"},
                            ],
                        }
                    ]
                    for _ in valid_images
                ]

                texts = [
                    self._processor.apply_chat_template(
                        conv, tokenize=False, add_generation_prompt=True
                    )
                    for conv in conversations
                ]

                inputs = self._processor(
                    text=texts,
                    images=list(valid_images),
                    return_tensors="pt",
                    padding=True,
                    do_pad=True,
                ).to(self.device)

                output_ids = cast(Any, self._model).generate(
                    **inputs,
                    max_new_tokens=self._model_max_length,
                    use_cache=True,
                )

                # Decode only generated tokens (strip input prompt)
                output_texts = [
                    self._processor.decode(
                        output_ids[i, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    for i in range(len(valid_images))
                ]

                for cluster, raw_text in zip(valid_clusters, output_texts):
                    _log.debug(
                        f"GraniteVision table [{cluster.id}] raw output: {raw_text!r}"
                    )
                    try:
                        otsl_seq, table_cells, num_rows, num_cols = _parse_otsl_output(
                            raw_text
                        )
                    except Exception as exc:
                        _log.warning(
                            f"Failed to parse OTSL output for table cluster {cluster.id}: {exc}"
                        )
                        otsl_seq, table_cells, num_rows, num_cols = [], [], 0, 0

                    tbl = Table(
                        otsl_seq=otsl_seq,
                        table_cells=table_cells,
                        num_rows=num_rows,
                        num_cols=num_cols,
                        id=cluster.id,
                        page_no=page.page_no,
                        cluster=cluster,
                        label=cluster.label,
                    )
                    table_prediction.table_map[cluster.id] = tbl

                predictions.append(table_prediction)

        return predictions
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_table_structure_granite_vision.py -v
```

Expected: all tests `PASSED` (the disabled/options tests don't load weights).

- [ ] **Step 5: Commit**

```bash
git add docling/models/stages/table_structure/table_structure_model_granite_vision.py \
        tests/test_table_structure_granite_vision.py
git commit -m "feat: implement GraniteVisionTableStructureModel"
```

---

## Task 4: Register the new engine

**Files:**
- Modify: `docling/models/plugins/defaults.py:61-74`
- Modify: `tests/test_table_structure_granite_vision.py`

- [ ] **Step 1: Write failing test for factory registration**

Add to `tests/test_table_structure_granite_vision.py`:

```python
def test_factory_registration():
    """GraniteVisionTableStructureModel must be discoverable via the factory."""
    from docling.models.factories.table_factory import TableStructureFactory
    from docling.models.stages.table_structure.table_structure_model_granite_vision import (
        GraniteVisionTableStructureModel,
    )

    factory = TableStructureFactory()
    engines = factory.get_all_engines()  # returns list of registered classes
    engine_types = [type(e) if not isinstance(e, type) else e for e in engines]
    assert GraniteVisionTableStructureModel in engine_types
```

> **Note:** Check how `TableStructureFactory` / `BaseFactory` exposes its registry before writing this test — read `docling/models/factories/base_factory.py` to find the right method name. Adjust `factory.get_all_engines()` to the actual API.

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest tests/test_table_structure_granite_vision.py::test_factory_registration -v
```

Expected: `AssertionError` — `GraniteVisionTableStructureModel` not yet registered.

- [ ] **Step 3: Read `base_factory.py` to understand the registry API**

```bash
cat docling/models/factories/base_factory.py
```

Adjust the test in Step 1 to use the actual method that lists registered engines, then re-run Step 2.

- [ ] **Step 4: Register in `defaults.py`**

Edit `docling/models/plugins/defaults.py`, updating `table_structure_engines()`:

```python
def table_structure_engines():
    from docling.models.stages.table_structure.table_structure_model import (
        TableStructureModel,
    )
    from docling.models.stages.table_structure.table_structure_model_granite_vision import (
        GraniteVisionTableStructureModel,
    )
    from docling.models.stages.table_structure.table_structure_model_v2 import (
        TableStructureModelV2,
    )

    return {
        "table_structure_engines": [
            TableStructureModel,
            TableStructureModelV2,
            GraniteVisionTableStructureModel,
        ]
    }
```

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/test_table_structure_granite_vision.py -v
```

Expected: all tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add docling/models/plugins/defaults.py tests/test_table_structure_granite_vision.py
git commit -m "feat: register GraniteVisionTableStructureModel in table_structure_engines"
```

---

## Task 5: Run the full test suite

- [ ] **Step 1: Run existing docling tests to check for regressions**

```bash
python -m pytest tests/ -v --ignore=tests/test_table_structure_granite_vision.py -x
```

Expected: all existing tests pass. If any fail, investigate whether the `pipeline_options.py` change broke discriminated-union resolution for `table_structure_options`.

- [ ] **Step 2: If failures relate to options discriminator, check the union type**

Search for where `BaseTableStructureOptions` is used in a `Union` / `Annotated` discriminator:

```bash
grep -n "BaseTableStructureOptions\|table_structure_options" docling/datamodel/pipeline_options.py | head -30
```

If `GraniteVisionTableStructureOptions` needs to be added to a `Union` type annotation for the discriminated union to work, add it there.

- [ ] **Step 3: Commit any fixes**

```bash
git add -p
git commit -m "fix: include GraniteVisionTableStructureOptions in pipeline options union"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - New model class → Task 3 ✓
  - OTSL parser with inline text → Task 2 ✓
  - `GraniteVisionTableStructureOptions` → Task 1 ✓
  - Model loading (same as `ChartExtractionModelGraniteVisionV4`) → Task 3 ✓
  - Cropped table image input → Task 3 ✓
  - Batched inference per page → Task 3 ✓
  - `bbox=None` for all cells → tested in Task 2 ✓
  - Error handling on parse failure → Task 3 ✓
  - Factory registration → Task 4 ✓
  - No MPS support → Task 3 (`supported_devices=[CPU, CUDA]`) ✓

- [x] **Placeholder scan:** No TBDs. Task 4 Step 3 has an explicit note to read `base_factory.py` before finalizing the test — intentional, not a gap.

- [x] **Type consistency:** `_parse_otsl_output` returns `tuple[list[str], list[TableCell], int, int]` in Task 2 and is consumed with `otsl_seq, table_cells, num_rows, num_cols = _parse_otsl_output(...)` in Task 3. ✓
