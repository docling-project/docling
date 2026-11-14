# GitHub conversations

Convert GitHub issues, pull requests, and discussions from JSON or retrieve them
through the authenticated GitHub CLI (`gh`). The backend produces a structured
`DoclingDocument` that can be exported to DCLX with image assets.

Install `docling` or `docling-slim[format-web,cli]`. For URL retrieval, install
`gh` and run `gh auth login` first.

## CLI

```bash
docling convert https://github.com/docling-project/docling-core/pull/1019 \
  --from json_github --to dclx --output ./output
```

GitHub conversation URLs also use this route when `--from` is omitted. Use
`--from html` to convert the web page instead. The native route follows all
requested comments, reviews, review threads, and discussion replies. Nested
connections are paginated separately. Retrieval errors abort the conversion.
The total retrieval timeout defaults to 60 seconds; change it with
`--github-timeout 180`.

Convert an existing export with the same backend:

```bash
gh pr view https://github.com/docling-project/docling-core/pull/1019 \
  --json url,number,title,body,author,state,createdAt,updatedAt,labels,comments,reviews > pr.json
docling convert pr.json --from json_github --to dclx --output ./output
```

`gh pr view` exports contain only the selected fields and do not include complete
inline review threads. Use the native URL route for complete retrieval. Local
JSON can also contain REST objects with associated comments, GraphQL responses
under `data.repository` or `data.node`, or the versioned SDK export envelope.
Native objects must include `title`, `number`, and a canonical web `url` or
`html_url`. Input is limited to 32 MiB. One object produces one document.

## Images

Image fetching follows the existing local and remote access controls:

```bash
docling convert pr.json --from json_github --to dclx \
  --github-image-fetch local --output ./output
docling convert https://github.com/docling-project/docling-core/pull/1019 \
  --from json_github --to dclx --github-image-fetch remote --output ./output
```

`--github-image-fetch` accepts `none` (default), `local`, `remote`, or `all`.
Embedded data images can be loaded without remote access. Local paths are
resolved against the JSON file directory. Remote images use the shared image
loader and its download and pixel limits. The `gh` authentication token is not
forwarded to image hosts. Private images that require separate authentication
may remain unavailable.

Missing images retain their picture position, description, and original
Markdown in source metadata. Warnings identify failed loads. Add
`--github-strict-assets` to fail when a picture cannot be loaded. Videos and
other linked attachments remain links. DCLX packages the loaded images.

## SDK

```python
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.utils.github import fetch_github_conversation, github_document_stream

source = fetch_github_conversation(
    "https://github.com/docling-project/docling-core/pull/1019", timeout=180
)
Path("pr-source.json").write_text(source.model_dump_json(indent=2), encoding="utf-8")

converter = DocumentConverter(allowed_formats=[InputFormat.JSON_GITHUB])
document = converter.convert(github_document_stream(source)).document
document.save_as_doclang_archive(Path("pr.dclx"), add_named_groups=True)
```

For local files, call `converter.convert(Path("pr-source.json"))`.
`GitHubJSONFormatOption(backend_options=GitHubJSONBackendOptions(...))` exposes
the Markdown and image options, including `enable_local_fetch`,
`enable_remote_fetch`, and `strict_assets`. The CLI enables named groups for
GitHub DCLX output. Set `add_named_groups=True` in SDK exports to retain entry
and reply ownership.

## Content and metadata

The document contains a title, source information, labels, the body, and the
conversation. Entry groups retain reply parents. Each entry has its author,
timestamps, permalink, and available review context. Accepted answers are
marked explicitly. No page coordinates are invented.

Issue and pull request labels appear in a **Labels** table after source
information and before the body. Its columns contain the original name and
description. A supplied label URL stays on its name. An omitted label collection
shows `Not supplied`; an empty collection shows `None`. Duplicate IDs are removed
in source order. When an ID is absent, the exact name is the identity.

The title's `meta.github__source` contains the validated source fields, labels,
IDs, colors, and original body. Entry headings use `meta.github__entry` for
their source fields and original Markdown. GitHub labels are separate from
`DocItemLabel`. JSON preserves these structured metadata types. DCLX retains the
visible labels, descriptions, hyperlinks, and named groups. Its custom metadata
can normalize scalar types and single-item collections; keep the source JSON
when exact source metadata types are required.

Code blocks retain parser-provided indentation, blank lines, and trailing
whitespace. Inline code follows Markdown whitespace rules. Language hints map
to Docling's language labels; unsupported hints remain in the original Markdown
stored in metadata. Diff excerpts remain code. Fenced HTML remains literal code.
Embedded HTML is parsed without script execution. Balanced `details`, `table`,
and `div` blocks can contain blank lines in nested preformatted code. Other HTML
uses the existing Markdown and HTML fallback, retaining supported structure and
readable content.

Link text and targets are retained. Fragment and root-relative links use the
conversation URL. Repository file links use the repository's default branch
when supplied or retrieved. When that context is absent, the relative target is
kept and a warning is emitted. Conversion does not fetch link destinations.

Custom metadata can appear in text serializers. To omit the source dictionaries
from an SDK Markdown export, pass
`blocked_meta_names={"github__source", "github__entry", "github__original_link", "github__language"}`
to `document.export_to_markdown()`.
