from enum import Enum
from pathlib import Path, PurePath
from typing import Annotated, Literal, Optional, Union

from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
    SecretStr,
    conint,
    model_validator,
)


class BaseBackendOptions(BaseModel):
    """Common options for all declarative document backends."""

    enable_remote_fetch: bool = Field(
        False, description="Enable remote resource fetching."
    )
    enable_local_fetch: bool = Field(
        False, description="Enable local resource fetching."
    )


class DeclarativeBackendOptions(BaseBackendOptions):
    """Default backend options for a declarative document backend."""

    kind: Literal["declarative"] = Field("declarative", exclude=True, repr=False)


class HTMLBackendOptions(BaseBackendOptions):
    """Options specific to the HTML backend.

    This class can be extended to include options specific to HTML processing.
    """

    kind: Literal["html"] = Field("html", exclude=True, repr=False)
    render_page: bool = Field(
        False,
        description=(
            "Render HTML in a headless browser to capture page images and "
            "element bounding boxes."
        ),
    )
    render_page_width: int = Field(
        794, description="Render page width in CSS pixels (A4 @ 96 DPI)."
    )
    render_page_height: int = Field(
        1123, description="Render page height in CSS pixels (A4 @ 96 DPI)."
    )
    render_page_orientation: Literal["portrait", "landscape"] = Field(
        "portrait", description="Render page orientation."
    )
    render_print_media: bool = Field(
        True, description="Use print media emulation when rendering."
    )
    render_wait_until: Literal["load", "domcontentloaded", "networkidle"] = Field(
        "networkidle",
        description="Playwright wait condition before extracting the DOM.",
    )
    render_wait_ms: int = Field(
        0, description="Extra delay in milliseconds after load."
    )
    render_device_scale: float = Field(
        1.0, description="Device scale factor for rendering."
    )
    page_padding: int = Field(
        0,
        description=(
            "Padding in CSS pixels applied to the HTML body before rendering."
        ),
    )
    render_full_page: bool = Field(
        False,
        description=("Capture a single full-height page image instead of paginating."),
    )
    render_dpi: int = Field(
        96, description="DPI used for page images created from rendering."
    )
    fetch_images: bool = Field(
        False,
        description=(
            "Whether the backend should access remote or local resources to parse "
            "images in an HTML document."
        ),
    )
    source_uri: Optional[Union[AnyUrl, PurePath]] = Field(
        None,
        description=(
            "The URI that originates the HTML document. If provided, the backend "
            "will use it to resolve relative paths in the HTML document."
        ),
    )
    headers: Annotated[
        dict[str, str] | None,
        Field(
            description=(
                "HTTP headers to include when fetching remote images. Use for "
                "authentication (e.g., API keys, bearer tokens) or custom headers "
                "required by image servers."
            ),
            examples=[{"Authorization": "Bearer TOKEN"}, {"X-API-Key": "your-api-key"}],
            repr=False,
        ),
    ] = None
    add_title: bool = Field(
        True, description="Add the HTML title tag as furniture in the DoclingDocument."
    )
    infer_furniture: bool = Field(
        True, description="Infer all the content before the first header as furniture."
    )
    max_image_data_base64_bytes: PositiveInt = Field(
        20 * 1024 * 1024,  # 20 MB
        description="The maximum number of base64 data bytes that the backend will accept.",
    )
    max_remote_image_bytes: PositiveInt = Field(
        20 * 1024 * 1024,  # 20 MB
        description="The maximum number of bytes for remote image downloads.",
    )
    max_redirects: Annotated[int, Field(ge=0)] = Field(
        5,
        description="Maximum number of HTTP redirects to follow when fetching remote resources. Set to 0 to disable redirects.",
    )


class MarkdownBackendOptions(BaseBackendOptions):
    """Options specific to the Markdown backend."""

    kind: Literal["md"] = Field("md", exclude=True, repr=False)
    fetch_images: bool = Field(
        False,
        description=(
            "Whether the backend should access remote or local resources to parse "
            "images in the markdown document."
        ),
    )
    source_uri: Optional[Union[AnyUrl, PurePath]] = Field(
        None,
        description=(
            "The URI that originates the markdown document. If provided, the backend "
            "will use it to resolve relative paths in the markdown document."
        ),
    )
    max_image_data_base64_bytes: PositiveInt = Field(
        20 * 1024 * 1024,  # 20 MB
        description="The maximum number of base64 data bytes that the backend will accept.",
    )


class EpubBackendOptions(BaseBackendOptions):
    """Options specific to the EPUB backend."""

    kind: Annotated[Literal["epub"], Field(exclude=True, repr=False)] = "epub"
    fetch_images: Annotated[
        bool, Field(description="Whether to fetch and process images from the EPUB.")
    ] = False
    max_total_bytes: Annotated[
        PositiveInt,
        Field(
            description="Maximum cumulative size in bytes of all data extracted from the EPUB archive during processing"
        ),
    ] = 100 * 1024 * 1024  # 100 MB
    max_file_bytes: Annotated[
        PositiveInt,
        Field(
            description="Maximum size in bytes for any single file extracted from the EPUB archive"
        ),
    ] = 10 * 1024 * 1024  # 10 MB
    max_member_count: Annotated[
        PositiveInt, Field(description="Maximum number of archive members to process")
    ] = 1000


class PdfBackendOptions(BaseBackendOptions):
    """Backend options for pdf document backends."""

    kind: Literal["pdf"] = Field("pdf", exclude=True, repr=False)
    password: Optional[SecretStr] = None
    enforce_same_font: bool = Field(
        True,
        description=(
            "Whether docling-parse should split text cells at font boundaries. "
            "Disable this when PDFs use separate fonts for base glyphs and "
            "diacritics that should remain in the same text cell."
        ),
    )


class ThreadedDoclingParseBackendOptions(PdfBackendOptions):
    """Options specific to the threaded docling-parse backend."""

    kind: Literal["threaded-docling-parse"] = Field(
        "threaded-docling-parse", exclude=True, repr=False
    )
    parser_threads: Optional[PositiveInt] = Field(
        None,
        description=(
            "Number of parser threads to use for the threaded docling-parse backend. "
            "If unset, the backend falls back to global accelerator thread settings."
        ),
    )
    release_native_memory_every_n_pages: conint(ge=0) = Field(
        128,
        description=(
            "Release native parser memory after every N decoded pages in the "
            "threaded docling-parse backend. Set to 0 to disable native-memory "
            "release."
        ),
    )


class MetsGbsBackendOptions(PdfBackendOptions):
    """Options specific to the METS-GBS document backend."""

    kind: Annotated[Literal["mets-gbs"], Field(exclude=True, repr=False)] = "mets-gbs"  # type: ignore[assignment]
    max_total_bytes: Annotated[
        PositiveInt,
        Field(
            description="Maximum cumulative size in bytes of all data extracted from the archive during processing"
        ),
    ] = 300 * 1024 * 1024
    max_file_bytes: Annotated[
        PositiveInt,
        Field(
            description="Maximum size in bytes for any single file extracted from the archive"
        ),
    ] = 10 * 1024 * 1024
    max_member_count: Annotated[
        PositiveInt, Field(description="Maximum number of archive members to process")
    ] = 1000


class IWorkBackendOptions(BaseBackendOptions):
    """Options specific to the Apple iWork document backends."""

    kind: Annotated[Literal["iwork"], Field(exclude=True, repr=False)] = "iwork"
    max_total_bytes: Annotated[
        PositiveInt,
        Field(
            description="Maximum cumulative size in bytes of all data read from the iWork archive during processing"
        ),
    ] = 300 * 1024 * 1024
    max_file_bytes: Annotated[
        PositiveInt,
        Field(
            description="Maximum size in bytes for any single member read from the iWork archive"
        ),
    ] = 100 * 1024 * 1024
    max_member_count: Annotated[
        PositiveInt, Field(description="Maximum number of archive members to inspect")
    ] = 5000


class MsExcelBackendOptions(BaseBackendOptions):
    """Options specific to the MS Excel backend."""

    kind: Literal["xlsx"] = Field("xlsx", exclude=True, repr=False)
    treat_singleton_as_text: bool = Field(
        False,
        description=(
            "Whether to treat singleton cells (1x1 tables with empty neighboring "
            "cells) as TextItem instead of TableItem."
        ),
    )

    parse_charts: bool = Field(
        True,
        description=(
            "Whether to parse native charts embedded in worksheets and chart "
            "sheets. Each chart becomes a PictureItem classified by chart type "
            "(bar, line, pie, scatter) and carrying the chart's underlying data "
            "reconstructed as a table. Set to False to skip chart parsing."
        ),
    )

    render_chart_images: bool = Field(
        False,
        description=(
            "Whether to render an image for each native chart and attach it to "
            "the chart PictureItem. The chart is isolated into a temporary "
            "workbook and rasterized with LibreOffice, the same external tool "
            "used for EMF/WMF images. Opt-in (default False) because it "
            "requires a LibreOffice installation and inflates the output size. "
            "Only takes effect when parse_charts is True."
        ),
    )

    gap_tolerance: int = Field(
        0,
        description=(
            "The tolerance (in number of empty rows/columns) for merging nearby "
            "data clusters into a single table. Default is 0 (strict)."
        ),
    )
    sheet_names: Optional[list[str]] = Field(
        None,
        description=(
            "An optional list of sheet names to include in conversion. "
            "When set, only sheets whose names appear in this list will be processed. "
            "Sheet names are matched case-sensitively. "
            "Set to None (default) to include all sheets."
        ),
    )


class MsPowerpointBackendOptions(BaseBackendOptions):
    """Options specific to the MS PowerPoint backend."""

    kind: Literal["pptx"] = Field("pptx", exclude=True, repr=False)

    render_chart_images: bool = Field(
        False,
        description=(
            "Whether to render an image for each native chart and attach it to "
            "the chart PictureItem. The chart's slide is isolated into a "
            "temporary presentation and rasterized with LibreOffice, the same "
            "external tool used for EMF/WMF images. Opt-in (default False) "
            "because it requires a LibreOffice installation and inflates the "
            "output size. Charts always keep their classification and "
            "reconstructed tabular data regardless of this option."
        ),
    )


class MsWordBackendOptions(BaseBackendOptions):
    """Options specific to the MS Word backend."""

    kind: Literal["docx"] = Field("docx", exclude=True, repr=False)

    render_chart_images: bool = Field(
        False,
        description=(
            "Whether to render an image for each native chart and attach it to "
            "the chart PictureItem. The chart drawing is isolated into a "
            "temporary document and rasterized with LibreOffice, the same "
            "external tool used for EMF/WMF images. Opt-in (default False) "
            "because it requires a LibreOffice installation and inflates the "
            "output size. Charts always keep their classification and "
            "reconstructed tabular data regardless of this option."
        ),
    )


class OdsBackendOptions(BaseBackendOptions):
    """Options specific to the ODS (OpenDocument Spreadsheet) backend."""

    kind: Annotated[Literal["ods"], Field(exclude=True, repr=False)] = "ods"
    treat_singleton_as_text: Annotated[
        bool,
        Field(
            description=(
                "Whether to treat singleton cells (1x1 tables with empty neighboring "
                "cells) as TextItem instead of TableItem."
            )
        ),
    ] = False
    gap_tolerance: Annotated[
        int,
        Field(
            description=(
                "The tolerance (in number of empty rows/columns) for merging nearby "
                "data clusters into a single table. Default is 0 (strict)."
            )
        ),
    ] = 0
    sheet_names: Annotated[
        Optional[list[str]],
        Field(
            description=(
                "An optional list of sheet names to include in conversion. "
                "When set, only sheets whose names appear in this list will be processed. "
                "Sheet names are matched case-sensitively. "
                "Set to None (default) to include all sheets."
            )
        ),
    ] = None


class LatexBackendOptions(BaseBackendOptions):
    """Options specific to the LaTeX backend."""

    kind: Literal["latex"] = Field("latex", exclude=True, repr=False)
    parse_timeout: Optional[float] = Field(
        30.0,
        description=(
            "Maximum time allowed for parsing a LaTeX document. "
            "Set to None to disable the timeout. Defaults to 30 s."
        ),
    )
    tikz_engine: Optional[Literal["tectonic"]] = Field(
        None,
        description=(
            "The engine to use for rendering Tikz diagrams into images. "
            "Set to 'tectonic' to enable asynchronous image generation."
        ),
    )
    tikz_engine_timeout: float = Field(
        60.0,
        description="The timeout in seconds for rendering a single TikZ diagram.",
    )
    tikz_engine_allow_shell_escape: bool = Field(
        False,
        description=(
            "Allow Tectonic TikZ rendering to enable shell escape during "
            "compilation. Disabled by default for safer rendering of untrusted "
            "LaTeX."
        ),
    )


class EmailBackendOptions(BaseBackendOptions):
    """Options specific to the email backend (``.eml`` and ``.msg``)."""

    kind: Literal["email"] = Field("email", exclude=True, repr=False)
    list_attachments: bool = Field(
        False,
        description=(
            "Whether to append a list of the email's attachment filenames to "
            "the converted document. Only the attachment names (and content "
            "types when available) are listed; the attachments' binary content "
            "is never embedded. Opt-in (default False)."
        ),
    )
    process_attachments: bool = Field(
        False,
        description=(
            "Whether to parse the email's attachments with Docling and embed "
            "their converted content under an 'Attachments' section, each "
            "attachment as its own subsection. Every supported format is routed "
            "through Docling's own DocumentConverter, so PDF/DOCX/PPTX/XLSX/"
            "images/HTML/... are handled the same way as standalone documents. "
            "Attachments whose format is unsupported, whose backend dependency "
            "is missing, or that fail to convert fall back to a name/type note. "
            "Opt-in (default False) because it runs the full conversion pipeline "
            "per attachment. Takes precedence over list_attachments."
        ),
    )
    max_attachments: int = Field(
        32,
        ge=0,
        description=(
            "Maximum number of attachments to process when process_attachments "
            "is enabled. Attachments beyond this count are ignored."
        ),
    )
    max_attachment_bytes: int = Field(
        25 * 1024 * 1024,  # 25 MB
        ge=0,
        description=(
            "Maximum size in bytes of a single attachment to process when "
            "process_attachments is enabled. Larger attachments are skipped with "
            "a note instead of being converted."
        ),
    )
    max_attachment_total_bytes: int = Field(
        100 * 1024 * 1024,  # 100 MB
        ge=0,
        description=(
            "Maximum cumulative size in bytes of all attachments processed for a "
            "single email when process_attachments is enabled. Once the budget is "
            "exhausted, remaining attachments are skipped with a note."
        ),
    )


class XBRLBackendOptions(BaseBackendOptions):
    """Options specific to the XBRL backend."""

    kind: Annotated[Literal["xbrl"], Field("xbrl", exclude=True, repr=False)] = "xbrl"
    taxonomy: Annotated[
        Path | None,
        Field(
            description=(
                "Path to a folder with the taxonomy required by the XBRL instance"
                " reports. It should include schemas (`.xsd`) and linkbases (`.xml`)"
                " referenced by the XBRL reports in their relative locations."
                " Optionally, it can also include taxonomy packages (`.zip`)"
                " referenced by the reports with absolute URLs and mapped to files"
                " with a taxonomy catalog (`catalog.xml`) for offline parsing."
            )
        ),
    ] = None


class EbcdicFieldType(str, Enum):
    """Storage type of a field inside an EBCDIC record.

    The names map to the COBOL usages found in a copybook: `string` is
    `USAGE DISPLAY` character data, `integer`/`unsigned_integer` are
    `COMP`/`BINARY`, `packed_decimal` is `COMP-3`, and `zoned_decimal` is
    signed `USAGE DISPLAY` numeric data. `skip` covers fillers whose bytes
    are consumed but never decoded.
    """

    STRING = "string"
    INTEGER = "integer"
    UNSIGNED_INTEGER = "unsigned_integer"
    PACKED_DECIMAL = "packed_decimal"
    ZONED_DECIMAL = "zoned_decimal"
    SKIP = "skip"


class EbcdicField(BaseModel):
    """A single fixed-width field of an EBCDIC record."""

    name: Annotated[str, Field(description="Column name of the decoded field.")]
    size: Annotated[PositiveInt, Field(description="Field width in bytes.")]
    type: Annotated[
        EbcdicFieldType, Field(description="How the bytes are decoded.")
    ] = EbcdicFieldType.STRING
    scale: Annotated[
        NonNegativeInt,
        Field(
            description=(
                "Number of implied decimal digits of a numeric field, i.e. the "
                "digits after `V` in the COBOL picture clause."
            )
        ),
    ] = 0


class EbcdicRecordLayout(BaseModel):
    """Field layout of one record schema, i.e. a single COBOL copybook."""

    fields: Annotated[
        list[EbcdicField],
        Field(min_length=1, description="Fields in physical record order."),
    ]
    name: Annotated[
        str, Field(description="Name of the schema, used as the table heading.")
    ] = "record"
    selector: Annotated[
        Optional[str],
        Field(
            description=(
                "Value of the record-type prefix that selects this schema. "
                "Required as soon as a layout defines more than one schema."
            )
        ),
    ] = None

    @property
    def size(self) -> int:
        """Length in bytes of a record following this layout."""
        return sum(item.size for item in self.fields)


class EbcdicLayout(BaseModel):
    """Parsing rules for an EBCDIC data file.

    A layout describes the bytes to ignore at the file boundaries, the optional
    prefix carried by every record, and the record schemas themselves. Records
    are fixed-length unless `record_length_field` declares a length prefix.
    """

    records: Annotated[
        list[EbcdicRecordLayout],
        Field(min_length=1, description="Record schemas present in the file."),
    ]
    description: Annotated[
        str, Field(description="Free-text description of the layout.")
    ] = ""
    header_size: Annotated[
        NonNegativeInt, Field(description="Bytes to skip at the start of the file.")
    ] = 0
    footer_size: Annotated[
        NonNegativeInt, Field(description="Bytes to skip at the end of the file.")
    ] = 0
    record_length_field: Annotated[
        Optional[EbcdicField],
        Field(
            description=(
                "Prefix field holding the total record length, prefix included. "
                "Set it for variable-length records; leave it unset when every "
                "record has the fixed size of its schema."
            )
        ),
    ] = None
    record_type_field: Annotated[
        Optional[EbcdicField],
        Field(
            description=(
                "Prefix field whose value selects the record schema. Required "
                "for multi-schema files."
            )
        ),
    ] = None

    @property
    def prefix_fields(self) -> list[EbcdicField]:
        """Prefix fields read ahead of every record, in physical order."""
        return [
            item
            for item in (self.record_length_field, self.record_type_field)
            if item is not None
        ]

    @property
    def prefix_size(self) -> int:
        """Length in bytes of the prefix read ahead of every record."""
        return sum(item.size for item in self.prefix_fields)

    def select(self, record_type: Optional[str]) -> Optional[EbcdicRecordLayout]:
        """Return the schema matching a record-type value, if any."""
        if self.record_type_field is None:
            return self.records[0]
        return next(
            (item for item in self.records if item.selector == record_type), None
        )

    @model_validator(mode="after")
    def _validate_records(self) -> "EbcdicLayout":
        if len(self.records) > 1 and self.record_type_field is None:
            raise ValueError(
                "record_type_field is required for a layout with several records"
            )
        if self.record_type_field is not None:
            selectors = [item.selector for item in self.records]
            if None in selectors:
                raise ValueError(
                    "every record needs a selector when record_type_field is set"
                )
            if len(set(selectors)) != len(selectors):
                raise ValueError("record selectors must be unique")
        return self


class EbcdicBackendOptions(BaseBackendOptions):
    """Options specific to the EBCDIC backend."""

    kind: Annotated[Literal["ebcdic"], Field(exclude=True, repr=False)] = "ebcdic"
    encoding: Annotated[
        str,
        Field(
            description=(
                "Python codec used to decode character data, e.g. `cp037` "
                "(US/Canada), `cp500` (international) or `cp1140` (euro)."
            )
        ),
    ] = "cp037"
    layout: Annotated[
        Optional[EbcdicLayout], Field(description="Parsing rules for the file.")
    ] = None
    layout_file: Annotated[
        Optional[Path],
        Field(description="Path to a JSON file holding the parsing rules."),
    ] = None
    max_records: Annotated[
        Optional[PositiveInt],
        Field(description="Stop after this many records. Unset reads the whole file."),
    ] = None
    strip_control_characters: Annotated[
        bool,
        Field(description="Drop control characters from decoded character data."),
    ] = True

    @model_validator(mode="after")
    def _validate_layout_source(self) -> "EbcdicBackendOptions":
        if self.layout is not None and self.layout_file is not None:
            raise ValueError("set either layout or layout_file, not both")
        return self


BackendOptions = Annotated[
    Union[
        DeclarativeBackendOptions,
        EbcdicBackendOptions,
        EpubBackendOptions,
        HTMLBackendOptions,
        MarkdownBackendOptions,
        PdfBackendOptions,
        ThreadedDoclingParseBackendOptions,
        MetsGbsBackendOptions,
        IWorkBackendOptions,
        MsExcelBackendOptions,
        MsPowerpointBackendOptions,
        MsWordBackendOptions,
        OdsBackendOptions,
        LatexBackendOptions,
        EmailBackendOptions,
        XBRLBackendOptions,
    ],
    Field(discriminator="kind"),
]
