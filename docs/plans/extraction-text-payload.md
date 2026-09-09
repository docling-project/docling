# Plan: multi-format, multi-channel extraction

> Status: proposal, to validate and refine. Follows on from
> [`extraction-engines.md`](extraction-engines.md), which added the remote
> engine and moved prompt style onto the model spec.

## Goal

Extend document extraction beyond today's image-only PDF/IMAGE path. The work
splits into **three separable dimensions**, each with its own defaults and user
options, that compose into one model request:

1. **Input format & source shape** — accept non-paginable, non-image formats
   (DOCX, HTML, MD) and formats that carry both structure and images (DCLX).
2. **Payload channels** — send the model page images, document text, or both.
3. **Page grouping & output granularity** — one model request per page, per
   chunk, or per whole document; bounded by the model's context limit and
   deciding the shape of the result.

`DocumentExtractor.extract()` / `extract_all()` and the default behavior for
PDF/IMAGE do not change. Everything below is additive, gated by options whose
defaults reproduce today's behavior.

## The dimensions at a glance

| Dimension | Axis values | First-cut default | Primary user option |
|-----------|-------------|-------------------|---------------------|
| 1. Format & source | image-paginable (PDF, IMAGE) · text-only (DOCX, HTML, MD) · structure+images (DCLX) | all registered | markdown serialization options (backends are internal) |
| 2. Channels | `IMAGE` · `TEXT` · `IMAGE_AND_TEXT` · `AUTO` | `AUTO` (image if available, else text) | `input_channels` |
| 3. Grouping | `PER_PAGE` · `CHUNKED` · `WHOLE_DOCUMENT` | **open — not ready** | `page_grouping` + context budget |

The three are orthogonal but **constrained**: dim 2 is bounded by what dim 1
offers (no page images for DOCX), and dim 3 interacts with the context budget
once dim 2 puts images and/or long text into a single request.

---

## Dimension 1 — Input format & source shape

Extraction opens each input into an `InputDocument` via a backend keyed on
`InputFormat` (`_get_default_extraction_option`). Formats differ by **what
channels they can offer**, which is the bridge to dimension 2:

| Format class | Formats | Backend | Offers images? | Offers text? |
|--------------|---------|---------|----------------|--------------|
| image-paginable | PDF, IMAGE | `PdfDocumentBackend` / image | ✅ rendered pages | ⚠️ via parse (not wired yet) |
| text-only, non-paginable | DOCX, HTML, MD | `DeclarativeDocumentBackend` | ✗ | ✅ `DoclingDocument` |
| structure + images | DCLX | `DocLangArchiveBackend` | ✅ restored from archive | ✅ `DoclingDocument` |

DCLX is the primary new format: `DocLangArchiveBackend.convert()` loads a full
`DoclingDocument` **with** `artifacts_dir`, i.e. exactly what a previous convert
produced — structured content *and* page images (`doc.pages[n].image`).

**What it touches.** `_get_default_extraction_option` gains DOCX/HTML/MD/DCLX
entries reusing the convert-side backends. The pipeline stops assuming a
`PdfDocumentBackend`: it normalizes any input to a **source view** exposing
page images and/or a `DoclingDocument`, from which channels are drawn.

**Backends are internal — the user only provides the file.** There is no
user-facing backend selection; the format is detected and its backend chosen
automatically, as in convert. `allowed_formats` still gates which formats run.

**User options — text serialization only.** The one thing the user can shape here
is how a document becomes the *text* channel:
- **Markdown input passes through as-is.** For `InputFormat.MD` the text channel
  is the raw file content — no `DoclingDocument` round-trip, no re-serialization.
- For formats that must be serialized (DOCX, HTML, DCLX), the text channel is
  produced from the `DoclingDocument`. Expose the existing docling-core markdown
  serialization options here (the `export_to_markdown` / serializer parameters:
  image placeholder handling, table mode, etc.) rather than a single boolean.
  Default matches convert's defaults.

**Open questions.** Which formats ship first (DOCX/HTML/MD/DCLX proposed; XML,
AsciiDoc, CSV are one entry each later). Which subset of the markdown
serialization options to surface, and whether they live on the pipeline options
or the model spec. Whether a non-markdown text form (plain text) is ever wanted.

---

## Dimension 2 — Payload channels

What actually goes to the model, per request, is an **ordered list of content
items** — the NuExtract-native shape. Three real payloads:

- **image only** — today's behavior (PDF/IMAGE).
- **text only** — new; the only option for DOCX/HTML/MD.
- **image + text** — new; natural for DCLX (and, via dim-1 parse, for PDF).

The card confirms all three: *"Multimodal inputs: text, images, or text +
images."* An image+text page is `content: [{"type":"image",…},
{"type":"text",…}]`.

**User option.** `input_channels: ChannelSelection = AUTO`. The channel is
user-choosable; the default prefers the page image.

- `AUTO` (default): **page image if the format has one, otherwise text.** PDF /
  IMAGE / DCLX → image; DOCX / HTML / MD → text. Combined image+text is *not*
  chosen automatically. Reproduces today's PDF behavior.
- `IMAGE` / `TEXT`: force a single channel. Requesting a channel a format cannot
  provide is a loud error (e.g. `IMAGE` on DOCX), not a silent drop — matches the
  "no attribute-probing" house rule.
- `IMAGE_AND_TEXT`: **explicit opt-in** to send both. Only meaningful for formats
  that offer both (DCLX; PDF once dim-1 parse text is wired). Never the default,
  because it roughly doubles context cost (see dim 3).

### Model interface: generalize to a content array

Today the only contract is `process_images(image_batch, prompt)`. Channels need
a payload that can hold image and/or text. Target shape:

```python
def process(self, requests: Iterable[list[ContentItem]], template: str)
    -> Iterable[VlmPrediction]
# ContentItem = ImageItem | TextItem
```

`process_images` stays as a thin adapter (image-only content) so convert-side
callers and the current image path are untouched. Keep `process` on the
**extraction models only**, behind a narrow protocol the pipeline checks; convert
engines do not get it.

### Remote API with multi-channel payloads — NuExtract carries the schema out-of-band

Critical finding from the card: the schema/template is **not** in the message
content — content holds only the document item(s); the template rides a separate
channel. Symmetric across transport:

| | Local (transformers) | Remote (vLLM / OpenAI-conformant) |
|---|---|---|
| Document | content item(s) `image` and/or `text` | same |
| Schema | `apply_chat_template(…, template=…)` | `extra_body.chat_template_kwargs.template` |

So the pipeline's existing `prompt`-as-serialized-template channel needs no
change for NuExtract; only transport differs. The shipped remote path
(`GRANITE_VISION_4_1_API` → `ApiVlmModel` → `api_image_request`) is the wrong
shape — it puts the prompt in the message text plus a base64 image. NuExtract
remote needs a sibling **`api_nuextract_request(content_items, template, url, …)`**
that builds the content array (any mix of image/text) and the top-level
`chat_template_kwargs`. vLLM merges `extra_body` into the top-level POST body,
which the existing `payload = {"messages": …, **params}` already accommodates.

Granite schema-instruction stays image-only: the card confirms it cannot take a
text payload, so it is absent from dims 2–3.

**Open questions.** Whether `ContentItem` is a new small model or we reuse an
existing docling content type. Whether image+text ordering (image-then-text) is
fixed or configurable.

---

## Dimension 3 — Page grouping & output granularity (OPEN — not ready for implementation)

How many pages go into one model request (`PER_PAGE` / `CHUNKED` /
`WHOLE_DOCUMENT`), which must **align with the model's context limit** and
**changes the output shape** (one result per page vs. one per chunk/document).
This dimension is **not designed yet** — do not implement it in the first cut.

**The one thing the first cut must respect now:** so that grouping can be added
later without breaking the API, decide up front **which input and output types
are arrays rather than singular**, even while they only ever hold one element
today:

- The model request payload is already an ordered `list[ContentItem]` — good,
  a multi-page request is just a longer list.
- The pipeline should carry requests and results as **collections** (a
  `list[ExtractedPageData]` already exists on `ExtractionResult`), so a future
  chunk/whole-doc result adds entries rather than changing the type.
- Anywhere the first cut is tempted to hardcode "one unit → one page", model it
  as a one-element sequence instead, and leave `page_no` semantics for the
  non-paginable / multi-page case explicitly undecided (candidate shapes:
  `page_span`, nullable `page_no`, or a document-level result alongside `pages`).

Everything else in this dimension — the `page_grouping` option, context budget
(`max_pages_per_request`, `max_input_tokens`, `on_context_overflow`), and the
final output-granularity contract — is deferred to a dedicated follow-up.

---

## How the dimensions compose

The pipeline becomes a small assembly line, each stage owning one dimension:

1. **Open** the input → a source view exposing page images and/or a
   `DoclingDocument` (dim 1).
2. **Select channels** per `input_channels`, validated against what the source
   offers (dim 2).
3. **Group** pages into requests, each an ordered `list[ContentItem]` (dim 3).
   First cut: one request per page (or one for a non-paginable doc) — but the
   request and result types are already collections, so grouping slots in later
   without an API change.
4. **Run** the model once per request via `process(requests, template)` — local
   builder or `api_nuextract_request` (dim 2 transport).
5. **Map** predictions back to a `list[ExtractedPageData]` (dim 3 output).

Steps 1, 2/4, and 3/5 are the three dimensions; each can be built and validated
without the others (e.g. dim 1 + text-only channel + per-page grouping is the
smallest end-to-end slice).

## User options (consolidated)

New fields on `VlmExtractionPipelineOptions`, all defaulting to today's behavior.
Backends are **not** user options (dim 1 is file-in, format detected).

| Option | Type | Default | Dimension |
|--------|------|---------|-----------|
| `input_channels` | `AUTO / IMAGE / TEXT / IMAGE_AND_TEXT` | `AUTO` (image if available, else text) | 2 |
| markdown serialization options | docling-core serializer params | convert defaults | 1 |
| `page_grouping` + context budget | — | — | 3 — **deferred, not designed** |

`IMAGE_AND_TEXT` is an explicit opt-in, never selected by `AUTO`.

## First cut vs deferred

Pick the lowest-risk value on each axis, then widen:

**First cut** — the text channel end to end for the text-only formats:
- Dim 1: register DOCX / HTML / MD (MD passes through as-is; DOCX/HTML serialized
  to markdown).
- Dim 2: the `TEXT` channel (which is what `AUTO` resolves to for these formats),
  local **and** remote NuExtract via `process` / `api_nuextract_request`.
- Dim 3: one request/result per document, carried as collections (single
  element), no `page_grouping` option yet.

**Deferred**
- **DCLX and its image channel.** DCLX is a primary format, but its `AUTO`
  default is the page image, which needs the image content-array path (page
  images pulled from the `DoclingDocument`). It lands with that work — the step
  right after the text slice — not in the text-only first cut.
- Dim 2: `IMAGE_AND_TEXT` (opt-in) for DCLX and PDF; image-payload remote
  NuExtract.
- Dim 3: the whole grouping/context-budget/output-granularity design.
- Dim 1: XML, AsciiDoc, CSV and other declarative formats (one entry each).
- Granite text mode — out; the model cannot take a text payload.

## Change set (by file)

1. **`document_extractor.py`** — register DOCX/HTML/MD in
   `_get_default_extraction_option` with their convert-side backends (DCLX joins
   with the image-channel work). Backends chosen automatically, not user-facing.
   (dim 1)
2. **`pipeline/extraction_vlm_pipeline.py`** — replace the `PdfDocumentBackend`
   assumption with the open → select → group → run → map assembly. `_extract_data`
   drives it; PDF/IMAGE + `AUTO` + `PER_PAGE` keeps today's output byte-for-byte.
   (dims 1–3)
3. **`models/extraction/`** — add `process(requests, template)` to the extraction
   models behind a narrow protocol. Local NuExtract: a `build_nuextract_inputs`
   generalization emitting image and/or text content items. (dim 2)
4. **`utils/api_nuextract_request.py`** (or extend `api_image_request.py`) —
   NuExtract-shaped remote request: content array + `chat_template_kwargs`. (dim 2)
5. **`datamodel/pipeline_options.py`** — `input_channels` (dim 2) and the
   markdown serialization options (dim 1), backward-compatible defaults. No
   `page_grouping` option yet (dim 3 deferred).
6. **`datamodel/vlm_model_specs.py`** — `NU_EXTRACT_API`
   (`ApiExtractionVlmOptions`, `NUEXTRACT` style); route `NUEXTRACT`-style API
   specs to the NuExtract request path, not plain `ApiVlmModel`. (dim 2)

## Backward compatibility

- Public API unchanged; new behavior is opt-in via options that default to
  today's values.
- `process_images` remains as an image-only adapter over `process`.
- PDF/IMAGE with defaults (`AUTO`→image, `PER_PAGE`) produce identical results.

## Testing

- Dim 1: a DOCX/MD fixture opens and yields the expected text via the declarative
  backend; a DCLX fixture exposes both text and page images.
- Dim 2 local: NuExtract text-only and image+text content arrays build correctly.
- Dim 2 remote: the POST body carries document item(s) in `messages[…].content`
  and the template under `chat_template_kwargs`, with no image for text-only, and
  `enable_remote_services=False` raises `OperationNotAllowed`.
- Dim 3: first cut only — a non-paginable doc yields a single-element
  `list[ExtractedPageData]`; grouping/context tests come with the dim-3 follow-up.
- Regression: PDF/IMAGE extraction output unchanged.

## Docs / packaging

- `docling/.agents/skills/docling/references/extraction.md`: a "Formats,
  channels, and grouping" section — which formats, that images are optional, the
  channel/grouping options, and a remote NuExtract example.
- Slim extras: remote NuExtract text needs no torch and no qwen-vl-utils; confirm
  the API-extraction extra covers it in `slim-packaging.md`.

---

# Post-implementation refinements

> Status: proposals raised after the first cut shipped, from a review of the
> usage pattern. Each is tracked with its own status and open decisions. Ordered
> by dependency, not priority: R2 and R4 are small and independent and can ship
> first; R1 is the deep change and R3 + R5-static fold into it.

## R1 — Adopt the modern stage model-spec style (the root smell)

**Problem.** Extraction uses the legacy `InlineVlmOptions` / `ApiVlmOptions`
shape (subclassed into `InlineExtractionVlmOptions` / `ApiExtractionVlmOptions`).
The modern style — `VlmModelSpec` + `StageModelPreset` + `StagePresetMixin`,
already adopted by convert's `VlmConvertOptions` — is not used at all.

Convert still *accepts* the legacy union for back-compat, but its presets are
modern. Extraction never touched the modern path, so it shares only the legacy
trait, not the upgrade.

**Why it matters (concrete, not tidiness).** The modern spec's `engine_options`
+ `AUTO_INLINE` is exactly what removes the `isinstance`-on-options-subclass
dispatch now hardcoded in `ExtractionVlmPipeline.__init__`
(`ApiExtractionVlmModel` vs `ApiVlmModel` vs `TransformersExtractionModel`).
Today extraction gets **no MLX / vLLM engine variants and no auto-inline
selection** — a user cannot run NuExtract on MLX even though the machinery
exists in `stage_model_specs.py`.

**Proposal.** Give extraction its own
`ExtractionVlmOptions(StagePresetMixin, VlmEngineOptionsMixin, BaseModel)` with
`model_spec: VlmModelSpec`, and register extraction presets (`nuextract_2b`,
`granite_vision_4_1`). The pipeline then selects the engine instead of branching
on the options class. `vlm_options` as the field name stays — it is symmetric
with convert; only the *type* changes.

**Open decisions.**
- Where do `extraction_prompt_style`, the `serialize_template` /
  `build_extraction_prompt` logic, and channel capability (R3) live? Options:
  (a) fields/methods on the extraction options subclass next to `model_spec`, or
  (b) carried in `StageModelPreset.stage_options`. Convert excludes per-stage
  prompt/response-format from the shared base spec because they vary per *stage*;
  for extraction these vary per *model*, which argues for (a).
- Do we keep the legacy `Inline/ApiExtractionVlmOptions` union accepted for
  back-compat (mirroring convert), or hard-cut to the preset style since
  extraction is new and has no external users yet?
- Scope: this is its own PR. Confirm it lands *after* R2/R4.

## R2 — Page range on the text channel (confirmed bug)

**Problem.** `_get_text_from_input` → `backend.convert()` →
`_serialize_doc(doc, page_no=None)` serializes the **entire** document; the image
paths honor `input_doc.limits.page_range`, the text path ignores it.

**Proposal.** In `_extract_via_text`, read `ext_res.input.limits.page_range` and
restrict serialization to it. `_serialize_doc` already supports a page set via
`MarkdownParams.pages`; the `params is None` branch loops the range through
`export_to_markdown(page_no=...)`. Skip the restriction for the MD raw-passthrough
(no pages exist there) and document that.

**Open decisions.**
- TEXT still collapses to one `page_no=1` result (the dim-3 grouping deferral).
  Restricting *content* to the range is independent and ships now; confirm we do
  not try to also emit per-page results here.

**Status.** Shipped. `_get_text_from_input` now reads
`input_doc.limits.page_range` and restricts serialization to the document pages
within it (default range serializes the whole document byte-for-byte; MD
raw-passthrough is unpaginated and unaffected). `_serialize_doc` takes a page
*set*. TEXT still collapses to a single `page_no=1` result (dim-3 deferral
unchanged).

## R3 — `AUTO` must depend on model capability, not just format

**Problem.** `_resolve_channel` computes `AUTO` from the **format only**
(`offers_image` / `offers_text`), then separately rejects text channels when
`style != NUEXTRACT`. Model capability ("can this model take text?") is a
**proxy** for the prompt style, not modeled. DOCX + a Granite spec → `AUTO`
picks TEXT → the style check then raises, blaming the prompt style rather than
the real cause.

**Proposal.** Declare capability on the spec — `accepts_text` / `accepts_image`
(or an `accepted_channels` set). Then `AUTO` = (what the format offers) ∩ (what
the model accepts), preferring image; forced channels validate against both sets
with capability-worded errors; `extraction_prompt_style != NUEXTRACT` stops
being a stand-in for "no text."

**Open decisions.**
- Two booleans vs. an `accepted_channels: set[ChannelSelection]` field.
- Folds into R1 (capability is a spec field). Confirm it does not ship
  standalone before R1.

## R4 — Stop restating the backend when overriding options

**Problem.** `ExtractionFormatOption.backend` is required (inherited from
`BaseFormatOption`), yet `_get_default_extraction_option` already holds the
canonical `format → backend` map. Overriding only `pipeline_options` forces the
user to re-type e.g. `DocLangArchiveBackend`.

**Proposal.** Make `backend` optional and resolve it in
`DocumentExtractor.__init__` where overrides are already merged per format: when
an override omits `backend`, fill it from `_get_default_extraction_option(fmt)`.
No new `with_options` helper (add only if callers want to clone *and* keep a
handle).

**Open decisions.**
- Relaxing `backend` to optional on `ExtractionFormatOption` vs. keeping it
  required on the base and resolving via a subclass validator. Prefer the former
  if it does not disturb the convert-side `FormatOption`.

**Status.** Shipped. `ExtractionFormatOption.backend` is now optional;
`DocumentExtractor.__init__` fills it from `_get_default_extraction_option(fmt)`
when an override omits it. No `with_options` helper added.

## R5 — Validation: split static (construction-time) from dynamic (per-document)

**Problem.** All validation is runtime, in `_resolve_channel`, surfaced
per-document. "IMAGE channel + markdown input" *is* caught (MD offers no image →
loud error) but only when `extract()` runs. `input_channels=IMAGE_AND_TEXT` with
a Granite spec is a **static** contradiction (options-only, no document needed)
yet only blows up mid-extraction.

**Proposal.** Split validation:
- **Static** (options-only: channel vs. model capability) → a pydantic
  `model_validator` at options/pipeline construction. Depends on R3's capability
  fields.
- **Dynamic** (channel vs. *this document's* format) → stays in
  `_resolve_channel`; format is per-document and cannot be checked earlier. Just
  reword its messages in capability terms.

**Open decisions.**
- The static half depends on R3; sequence it into the R1/R3 change. The reworded
  runtime messages can ship earlier if convenient.

## Suggested sequencing

1. **R2 + R4** — small, independent, no design decisions blocking them.
2. **R1** — the model-spec upgrade; resolve its open decisions first.
3. **R3 + R5-static** — fold into R1 (capability-on-spec is the shared enabler);
   R5's reworded runtime messages may land with R3.
