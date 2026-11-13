# GitHub JSON backend

Status: implemented; local validation complete

## Implementation record

- Added `InputFormat.JSON_GITHUB`, typed source models, the JSON backend, and
  `GitHubJSONFormatOption`.
- Added complete `gh api` retrieval and native GitHub URL handling in the CLI.
- Added code, HTML, link, label, image, thread, and reply preservation tests.
- Verified core PR #1019, core issue #1018, and cli/cli discussion #3993 through
  the CLI. Reloaded each DCLX archive and checked its named groups and links.
- Ran the affected conversion, input detection, and CLI tests: 510 passed,
  1 skipped. The 26 GitHub tests also pass after the accepted-answer fixture
  update. `make validate` passes with no hook changes.
- Added CLI and SDK instructions in `docs/usage/github.md` and a runnable
  example in `docs/examples/convert_github.py`.

The models live in `docling/datamodel/github.py` so retrieval, detection, and
conversion share one schema without a utility-to-backend dependency. The
backend uses a GitHub-specific Markdown extension for balanced HTML blocks.
Repository-relative links use the known default branch; unknown branch context
keeps the original target. SDK DCLX exports must enable `add_named_groups=True`.
The CLI does this automatically. Exact metadata types remain in source JSON;
DocLang custom metadata can normalize scalar and collection types.

## Objective

Add a backend in `docling/backend/json/` that converts GitHub issues,
pull requests, and discussions into `DoclingDocument`.

Support JSON from disk and JSON retrieved through `gh`.
Support DCLX export with available image assets.

Treat code snippets, HTML snippets, links, and GitHub labels as explicit
preservation requirements. Verify them through DCLX export and reload.

## Existing support

- `DoclingJSONBackend` loads saved `DoclingDocument` JSON.
- `MarkdownDocumentBackend` parses GitHub Flavored Markdown.
- `ImageResourceLoader` loads images with local and remote access controls.
- The SDK and CLI can export DCLX archives.

Keep these paths compatible.

## Input contract

Support these documented input shapes:

- `gh issue view --json ...`.
- `gh pr view --json ...`.
- GitHub REST issue and pull request objects, with associated comments.
- GitHub GraphQL issue, pull request, and discussion responses.
- A versioned export envelope that combines an object with its comments,
  reviews, review threads, replies, and retrieval status.

Use typed models to normalize these shapes into one conversation model.
Accept both REST field names and GraphQL field names.

Preserve the distinction between missing data and empty collections.
For example, an omitted comments field does not mean zero comments.

Convert one issue, pull request, or discussion into one document.
Repository exports and collections of unrelated objects are out of scope
for the first version.

## Architecture

### 1. JSON backend and models

Add `GitHubJSONBackend` in
`docling/backend/json/github_json_backend.py`.

Place the input models and normalization helpers in adjacent modules.
The backend accepts paths and streams. It does not invoke `gh`.

Add proposed public types:

- `InputFormat.JSON_GITHUB`.
- `GitHubJSONFormatOption`.
- `GitHubJSONBackendOptions`.

Use `SimplePipeline`.

### 2. Format detection

Register the new format in the converter and format maps.

Distinguish GitHub JSON from saved Docling JSON using validated object
structure and source identity. Do not classify arbitrary JSON by extension
or by the presence of a `title` field.

Define a bounded detection policy. Test objects with large bodies and
different field orders. Report ambiguous inputs with an actionable error.

Keep existing `JSON_DOCLING` detection and conversion behavior.

### 3. Retrieval through gh

Add a separate retrieval helper in `docling/utils/github.py`.

Accept an explicit GitHub issue, pull request, or discussion URL.
Use `gh api` for complete retrieval. Use GraphQL for discussions.

Retrieve:

- Issue body and comments.
- Pull request body, conversation comments, reviews, and review threads.
- Discussion body, comments, replies, and accepted answer.
- Labels and their available descriptions and source attributes.

Follow every required pagination cursor or REST page link.
Paginate nested reply and thread connections separately. Include label
connections when the API paginates them.

Return the same typed export envelope that can be saved to disk and
passed to the backend.

Use argument lists and subprocess timeouts. Use existing gh authentication.
Report missing gh, authentication failures, rate limits, and partial
retrieval clearly. Do not return truncated conversations as complete.

Document direct `gh issue view` and `gh pr view` exports as supported
inputs whose content depends on the requested fields.

### 4. DoclingDocument mapping

Map content as follows:

| GitHub content | Document content |
| --- | --- |
| Object title | Document title |
| Repository, type, number, URL, state | Source information section |
| Labels | Labels section after source information, before the body |
| Author and timestamps | Source and entry information |
| Body | Structured Markdown content |
| Comments | Ordered conversation sections |
| Reviews | Review sections with author and decision |
| Review threads | Nested sections with file and line context |
| Discussion replies | Nested reply sections |
| Accepted answer | Explicit answer marker |
| Images | Picture items with available image data |

Preserve stable IDs, URLs, parent relationships, assignees, review state,
and available timestamps through supported metadata fields.
Include essential source information in document content so it survives
exports that omit custom metadata.

Reuse Markdown parsing for each body. Keep each entry within its own
section. Do not concatenate raw Markdown before parsing.

Use deterministic ordering. Keep replies under their parent.
Retain file paths, line ranges, diff excerpts, and resolved or outdated
state when supplied.

Handle deleted authors, empty bodies, and unavailable optional fields.
Do not invent page coordinates for conversation content.

#### Code snippets

Preserve fenced and indented code blocks as code items. Preserve inline
code as inline code. Retain available language tags, indentation, blank
lines, line breaks, and literal characters after JSON decoding. Do not
strip code whitespace or parse Markdown inside a code snippet.

Treat fenced HTML and XML as literal code. Preserve angle brackets,
entities, backticks, and text that resembles links without interpreting
them as document markup. Apply the same rule to diff excerpts and patch
snippets. Do not add source line numbers to the code text.

Test nested fences, snippets within lists and quotes, Unicode, and code
containing HTML or Markdown delimiters. Define how unsupported language
tags are retained when the document model uses a fixed language enum.

#### HTML snippets

Distinguish literal HTML examples from embedded HTML used as content.
Fenced HTML and inline code must remain code. Parse embedded HTML through
the existing Markdown and HTML path without executing scripts.

Preserve supported block and inline structure, including tables, lists,
`pre` and `code`, links, and images. Test GitHub patterns such as
`details` and `summary`, mixed Markdown and HTML, and escaped HTML examples.
Decode entities according to the content context without double decoding.

Define and document a fallback for unsupported elements. Retain readable
content and link targets. Do not silently discard a section. Verify that
HTML conversion keeps its conversation parent and adjacent content order.

#### Links

Preserve both display text and target in document hyperlink fields.
Cover inline and reference-style Markdown links, autolinks, bare URLs,
HTML anchors, image links, and links inside tables and formatted text.

Keep query strings, fragments, and escaped characters. Resolve relative
targets against a documented source context while retaining the original
target in source metadata where needed. Distinguish conversation-relative
links from repository file links. Keep unresolved targets and report the
missing context instead of inventing a URL.

Keep source and comment permalinks as clickable links. Do not turn code
text into hyperlinks. Do not fetch a target merely because it is a link.

Verify link targets as well as visible text after DCLX reload. If the
current serializer loses hyperlinks, propose the required serializer
change before implementation. Do not accept plain text as equivalent.

#### GitHub labels

Use a typed label model in the normalized source data. Preserve `name`,
`description`, source ID, color, and URL when supplied. Keep original
label names and descriptions without paraphrasing or case changes.

Store labels at the issue or pull request level. Retain a distinction
between omitted labels and an explicitly empty label collection.
Do not infer labels from discussion categories or body text.

Create one visible `Labels` section directly after source information and
before the body. Use a table with `Name` and `Description` columns. Link
the name to the label URL when supplied. Use an empty description cell
when no description is available. Show `None` for an explicitly empty
collection and `Not supplied` when the input omits labels.

Keep all available label attributes in typed source metadata. Choose the
supported Docling metadata extension point during implementation. Do not
encode GitHub labels as `DocItemLabel` values or repeat them on comments.

Preserve source order and remove duplicate label identities. Keep distinct
labels with distinct source IDs. Document the fallback identity rule for
exports that supply names only.

Verify that names, descriptions, and label hyperlinks survive DCLX reload.
Document whether IDs and colors also survive metadata serialization.
The visible section must remain useful when custom metadata is omitted.

### 5. Assets and DCLX

Reuse the image loader and existing access options.
Resolve local assets against the input file directory.
Resolve remote references against the documented source context.

Load permitted image data into picture items before export.
Deduplicate repeated image resources.

Keep links and image descriptions when an asset cannot be loaded.
Report missing assets. Provide a strict option for callers that require
all referenced images.

Export through `save_as_doclang_archive()`.
Verify which group and metadata fields survive its default settings.
Use supported export options where needed. Document any remaining loss.
Require code content, supported HTML structure, link targets, and the
visible Labels section to survive export and reload.

The first version packages supported images. Keep videos, archives,
and other attachments as links. Record this limit in the documentation.

## Implementation sequence

1. Define representative fixtures and the supported input contract.
2. Verify code, HTML, hyperlinks, labels, and DCLX preservation contracts.
   Identify any required changes in docling-core before building on them.
3. Add typed normalization models and adapters.
4. Add the backend, options, converter registration, and detection.
5. Map conversation content, labels, and Markdown into DoclingDocument.
6. Add image handling and verify DCLX export.
7. Add complete retrieval through gh.
8. Add SDK and CLI examples and update usage documentation.

Each stage requires a proposed patch and approval before file changes.

## Validation

Use small, self-authored fixtures. Keep routine tests offline.

Test:

- Each supported object type and input shape.
- Equivalent conversion from a path and a stream.
- Existing Docling JSON conversion.
- Ambiguous, malformed, and unrelated JSON.
- Missing comments versus empty comments.
- Comment order, reply parents, review context, and accepted answers.
- Code text, indentation, line breaks, language tags, and inline code.
- Literal HTML code versus embedded HTML, including mixed content.
- Markdown and HTML links, relative targets, queries, and fragments.
- Label placement, names, descriptions, IDs, colors, URLs, and missing data.
- Markdown tables, lists, and images.
- Deleted authors and missing optional data.
- REST pagination and nested GraphQL pagination, including labels.
- gh failures and incomplete retrieval.
- Asset access settings and unavailable images.
- DCLX archive contents and reload.

For DCLX, verify conversation text, essential source information,
thread relationships, tables, exact code content, supported HTML
structure, hyperlink targets, labels, and image assets after reload.
Do not use archive creation alone as the success check.

Run affected tests and `make validate`.
Review hook changes and repeat validation until it passes.

## Documentation

Update:

- `docs/usage/supported_formats.md`.
- A new GitHub conversion example.
- The packaged Docling usage skill.

Show local JSON conversion, gh retrieval, image options, and DCLX export.
State the accepted input shapes and completeness limits. Explain code and
HTML handling, relative link resolution, and the location and schema of
GitHub labels. State which source attributes survive DCLX export.

## Acceptance criteria

- All three GitHub object types convert from disk and gh retrieval.
- Native supported JSON does not require manual reshaping.
- Retrieved conversations include all requested pages.
- Document structure preserves entries and parent relationships.
- Code snippets retain their content and formatting semantics.
- HTML examples remain code; embedded HTML retains supported structure.
- Links retain their display text and targets through DCLX reload.
- Labels have a defined metadata location and a visible section with
  original names and available descriptions.
- DCLX contains loaded images and retains essential conversation content.
- Existing Docling JSON behavior remains compatible.
- Targeted tests and repository validation pass.

## References

- https://cli.github.com/manual/gh_issue_view
- https://cli.github.com/manual/gh_pr_view
- https://cli.github.com/manual/gh_api
- https://docs.github.com/en/graphql/guides/using-the-graphql-api-for-discussions
