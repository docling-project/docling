from pathlib import Path, PurePath
from typing import Annotated, Literal, Optional, Union

from pydantic import AnyUrl, BaseModel, Field, PositiveInt, SecretStr, conint


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


class PdfBackendOptions(BaseBackendOptions):
    """Backend options for pdf document backends."""

    kind: Literal["pdf"] = Field("pdf", exclude=True, repr=False)
    password: Optional[SecretStr] = None


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


BackendOptions = Annotated[
    Union[
        DeclarativeBackendOptions,
        HTMLBackendOptions,
        MarkdownBackendOptions,
        PdfBackendOptions,
        MetsGbsBackendOptions,
        MsExcelBackendOptions,
        LatexBackendOptions,
        XBRLBackendOptions,
    ],
    Field(discriminator="kind"),
]
