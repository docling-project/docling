import logging
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from docling_core.types.doc import DoclingDocument, DocumentOrigin
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.backend_options import EpubBackendOptions, HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class EpubDocumentBackend(DeclarativeDocumentBackend):
    """Backend for converting EPUB files to DoclingDocument format.

    EPUB files are essentially ZIP archives containing XHTML content files.
    This backend extracts the content and delegates to HTMLDocumentBackend
    for processing the XHTML structure.

    Known Limitations:
        - Internal anchor links (e.g., footnote references) are converted but
          the target anchor IDs are not preserved in the final Markdown output.
          This is a limitation of the HTML-to-DoclingDocument conversion process.
          Links like [1](#note-1) will be present, but the corresponding anchor
          targets may not be accessible in the exported Markdown.
    """

    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: EpubBackendOptions | None = None,
    ):
        if options is None:
            options = EpubBackendOptions()
        super().__init__(in_doc, path_or_stream, options)

        _log.debug("Starting EpubDocumentBackend...")

        self.path_or_stream = path_or_stream
        self.valid = False
        self.epub_zip: ZipFile | None = None
        self.content_files: list[str] = []
        self.metadata: dict[str, str] = {}

        try:
            # Open the EPUB file as a ZIP archive
            if isinstance(self.path_or_stream, BytesIO):
                self.epub_zip = ZipFile(self.path_or_stream, "r")
            elif isinstance(self.path_or_stream, Path):
                self.epub_zip = ZipFile(self.path_or_stream, "r")
            else:
                raise ValueError("path_or_stream must be BytesIO or Path")

            # Parse the EPUB structure
            self._parse_epub_structure()
            self.valid = True

            _log.debug(f"Found {len(self.content_files)} content files in EPUB")
        except Exception as e:
            _log.error(f"Failed to initialize EPUB backend: {e}")
            raise RuntimeError(
                f"Could not initialize EPUB backend for file with hash {self.document_hash}."
            ) from e

    def _parse_epub_structure(self):
        """Parse the EPUB structure to find content files and metadata."""
        if not self.epub_zip:
            return

        # Read container.xml to find the content.opf file
        try:
            container_data = self.epub_zip.read("META-INF/container.xml")
            container_root = ET.fromstring(container_data)

            # Find the content.opf path
            ns = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
            rootfile = container_root.find(".//container:rootfile", ns)

            if rootfile is None:
                raise ValueError("Could not find rootfile in container.xml")

            opf_path = rootfile.get("full-path")
            if not opf_path:
                raise ValueError("Could not find full-path in rootfile")

            # Parse content.opf to get the reading order
            opf_data = self.epub_zip.read(opf_path)
            opf_root = ET.fromstring(opf_data)

            # Extract metadata
            self._extract_metadata(opf_root)

            # Get the base directory for content files
            opf_dir = str(Path(opf_path).parent)
            if opf_dir == ".":
                opf_dir = ""

            # Extract spine (reading order)
            ns_opf = {"opf": "http://www.idpf.org/2007/opf"}
            spine = opf_root.find(".//opf:spine", ns_opf)
            manifest = opf_root.find(".//opf:manifest", ns_opf)

            if spine is None or manifest is None:
                raise ValueError("Could not find spine or manifest in content.opf")

            # Build manifest map (id -> href)
            manifest_map = {}
            for item in manifest.findall("opf:item", ns_opf):
                item_id = item.get("id")
                href = item.get("href")
                if item_id and href:
                    manifest_map[item_id] = href

            # Get content files in reading order
            for itemref in spine.findall("opf:itemref", ns_opf):
                idref = itemref.get("idref")
                if idref and idref in manifest_map:
                    href = manifest_map[idref]
                    # Construct full path
                    if opf_dir:
                        full_path = f"{opf_dir}/{href}"
                    else:
                        full_path = href
                    self.content_files.append(full_path)

            _log.debug(f"Content files in reading order: {self.content_files}")

        except Exception as e:
            _log.error(f"Error parsing EPUB structure: {e}")
            raise

    def _extract_metadata(self, opf_root: ET.Element):
        """Extract metadata from the OPF file."""
        ns_dc = {"dc": "http://purl.org/dc/elements/1.1/"}
        ns_opf = {"opf": "http://www.idpf.org/2007/opf"}

        metadata = opf_root.find(".//opf:metadata", ns_opf)
        if metadata is None:
            return

        # Extract common metadata fields
        for field in [
            "title",
            "creator",
            "publisher",
            "date",
            "language",
            "identifier",
        ]:
            element = metadata.find(f"dc:{field}", ns_dc)
            if element is not None and element.text:
                self.metadata[field] = element.text

        _log.debug(f"Extracted metadata: {self.metadata}")

    def _fix_internal_links(self, html_content: str) -> str:
        """Fix internal links that reference other XHTML files.

        When combining multiple XHTML files into one HTML document, links like
        'endnotes.xhtml#note-1' need to be converted to '#note-1' since all
        content is now in a single file.

        Args:
            html_content: HTML content with potentially broken internal links

        Returns:
            HTML content with fixed internal links
        """
        # Pattern to match href attributes that reference .xhtml files with anchors
        # Examples: href="endnotes.xhtml#note-1" or href="chapter-1.xhtml#section-2"
        pattern = r'href="([^"]*\.xhtml)(#[^"]*)"'

        # Replace with just the anchor part
        fixed_content = re.sub(pattern, r'href="\2"', html_content)

        return fixed_content

    def is_valid(self) -> bool:
        return self.valid

    def unload(self):
        """Clean up resources."""
        if self.epub_zip:
            self.epub_zip.close()
            self.epub_zip = None
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()
        self.path_or_stream = None

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.EPUB}

    def convert(self) -> DoclingDocument:
        """Convert the EPUB file to a DoclingDocument.

        This method extracts all content files from the EPUB and processes
        them sequentially using the HTMLDocumentBackend.
        """
        _log.debug("Converting EPUB...")

        if not self.is_valid() or not self.epub_zip:
            raise RuntimeError(
                f"Cannot convert EPUB with hash {self.document_hash} because the backend failed to init."
            )

        # Create document origin
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/epub+zip",
            binary_hash=self.document_hash,
        )

        # Initialize the main document
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        # Concatenate all XHTML content into a single HTML document
        combined_html_parts = [
            '<!DOCTYPE html><html><head><meta charset="utf-8"/></head><body>'
        ]

        # TODO: leverage `self.metadata` once DoclingDocument supports file metadata

        # Process each content file in reading order
        for content_file in self.content_files:
            try:
                _log.debug(f"Reading content file: {content_file}")

                # Read the XHTML content
                xhtml_data = self.epub_zip.read(content_file)
                xhtml_text = xhtml_data.decode("utf-8")

                # Extract the body content from the XHTML
                # Simple extraction - find content between <body> tags
                body_match = re.search(
                    r"<body[^>]*>(.*?)</body>", xhtml_text, re.DOTALL | re.IGNORECASE
                )

                if body_match:
                    body_content = body_match.group(1)
                else:
                    # If no body tags, use the whole content
                    body_content = xhtml_text

                # Fix internal links: convert file.xhtml#anchor to #anchor
                # This is necessary because we're combining all XHTML files into one HTML
                body_content = self._fix_internal_links(body_content)

                combined_html_parts.append(body_content)

            except Exception as e:
                _log.warning(f"Failed to read content file {content_file}: {e}")
                continue

        combined_html_parts.append("</body></html>")
        combined_html = "\n".join(combined_html_parts)

        # Now process the combined HTML with HTMLDocumentBackend
        html_stream = BytesIO(combined_html.encode("utf-8"))

        epub_options = self.options
        html_options = HTMLBackendOptions(
            enable_local_fetch=epub_options.enable_local_fetch,
            enable_remote_fetch=epub_options.enable_remote_fetch,
            fetch_images=epub_options.fetch_images,
            infer_furniture=False,
            add_title=False,  # We already added the title
        )

        in_doc = InputDocument(
            path_or_stream=html_stream,
            format=InputFormat.HTML,
            backend=HTMLDocumentBackend,
            filename="combined.html",
            backend_options=html_options,
        )

        html_backend = HTMLDocumentBackend(
            in_doc=in_doc,
            path_or_stream=html_stream,
            options=html_options,
        )

        # Convert and return the combined document
        doc = html_backend.convert()

        # Update the origin to reflect it's an EPUB
        doc.origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/epub+zip",
            binary_hash=self.document_hash,
        )
        doc.name = self.file.stem or "file"

        html_stream.close()

        return doc
