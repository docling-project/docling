# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Backend to parse XBRL (eXtensible Business Reporting Language) documents.

XBRL is a standard XML format used for business and financial reporting.
It is widely used by companies, regulators, and financial institutions worldwide
for exchanging financial information.

This backend leverages the Arelle library for XBRL processing.

Warning:
    This implementation uses DoclingDocument's GraphData object to represent
    key-value pairs extracted from XBRL numeric facts. The design of key-value
    pairs (and therefore the GraphData, GraphCell, GraphLink class) may soon
    change in a new release of the `docling-core` library. This implementation
    will need to be updated accordingly when that release is available.
"""

from __future__ import annotations

import html
import logging
import re
import shutil
import threading
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final
from urllib.parse import unquote, urljoin, urlparse
from xml.sax.saxutils import escape as xml_escape

from docling_core.types.doc import (
    DoclingDocument,
    DocumentOrigin,
    GraphCell,
    GraphCellLabel,
    GraphData,
    GraphLink,
    GraphLinkLabel,
)
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.backend_options import HTMLBackendOptions, XBRLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError, OperationNotAllowed

_XBRL_AVAILABLE: bool = False
_XBRL_IMPORT_ERROR: ImportError | None = None
try:
    from arelle import (
        Cntlr,
        PackageManager as _ArellePackageManager,
    )
    from arelle.ModelDocument import Type
    from arelle.ModelDtsObject import ModelConcept
    from arelle.ModelXbrl import ModelXbrl

    from docling.backend.xml.xbrl_web_cache import FetchPolicyWebCache

    _XBRL_AVAILABLE = True
except ImportError as e:
    _XBRL_IMPORT_ERROR = e

_log = logging.getLogger(__name__)


_WEB_CACHE_TIMEOUT: Final[int] = 10
_ARELLE_LOAD_LOCK: Final = threading.Lock()


# Elements that reference schemas and linkbases of an XBRL or iXBRL document. The
# prefixes of the element and of the `href` attribute are not fixed by the
# specification. The tag body excludes `<` and `>`, so that the pattern stays
# linear on malformed input.
_REF_TAG_RE: Final = re.compile(
    rb"<(?:[\w.-]+:)?(?:schemaRef|linkbaseRef|roleRef|arcroleRef)\b[^<>]*>"
)
_HREF_ATTR_RE: Final = re.compile(
    rb"(?<![\w.:-])((?:[\w.-]+:)?href\s*=\s*)(?:\"([^\"]*)\"|'([^']*)')"
)


def _is_local_file(base_dir: Path, reference: str) -> bool:
    """Tell whether a reference points to an existing file inside ``base_dir``.

    A reference that leaves ``base_dir`` (``..`` segments or an absolute path)
    never counts as local, so that a document cannot probe the file system.
    """
    try:
        root = base_dir.resolve()
        target = (root / reference).resolve()
        return target.is_relative_to(root) and target.exists()
    except (OSError, ValueError):
        return False


def _absolutize_missing_refs(content: bytes, base_dir: Path, source_uri: str) -> bytes:
    """Make the relative references that are not found locally absolute.

    Relative references to schemas and linkbases are resolved against the URL
    where the document was published, unless the referenced file exists in
    ``base_dir`` (e.g. it was provided in the ``taxonomy`` folder). The result
    is escaped for XML, and it always stays on the origin of ``source_uri``. The fetch
    policy of the web cache still applies to every resulting URL.

    Args:
        content: The XBRL or iXBRL document.
        base_dir: The folder that holds the local copy of the document.
        source_uri: The http(s) URL where the document was published.

    Returns:
        The document with the missing relative references replaced.
    """

    def _replace_href(match: re.Match[bytes]) -> bytes:
        quote = b'"' if match.group(2) is not None else b"'"
        raw = match.group(2) if match.group(2) is not None else match.group(3)
        href = html.unescape(raw.decode("utf-8", errors="replace")).strip()
        path_part = href.split("#", 1)[0]
        if (
            not path_part
            or urlparse(href).scheme
            or _is_local_file(base_dir, unquote(path_part))
        ):
            return match.group(0)
        absolute = urljoin(source_uri, href)
        # A relative reference must stay on the origin of the source URL. This
        # drops, for instance, the network-path reference `//other.host/x.xsd`.
        origin, resolved = urlparse(source_uri), urlparse(absolute)
        if (resolved.scheme, resolved.netloc) != (origin.scheme, origin.netloc):
            return match.group(0)
        _log.debug(f"Resolving reference {href!r} to {absolute!r}")
        escaped = xml_escape(absolute, {'"': "&quot;", "'": "&apos;"})
        return match.group(1) + quote + escaped.encode("utf-8") + quote

    def _replace_tag(match: re.Match[bytes]) -> bytes:
        return _HREF_ATTR_RE.sub(_replace_href, match.group(0), count=1)

    return _REF_TAG_RE.sub(_replace_tag, content)


def _warn_unresolved_facts(model: ModelXbrl) -> None:
    """Warn when the facts of the instance cannot be linked to their concepts.

    Arelle does not raise an error when a schema or linkbase of the report is not
    found. The facts stay without a concept, and the numeric facts and text blocks
    are silently dropped from the converted document.
    """
    total = len(model.facts)
    unresolved = sum(1 for fact in model.facts if fact.concept is None)
    if unresolved:
        _log.warning(
            f"{unresolved} of {total} facts could not be linked to a concept, so"
            " they are not converted. The taxonomy of the report was not found."
            " Check the `taxonomy` folder (a taxonomy package with a catch-all"
            " catalog needs the extension taxonomy files next to it),"
            " `enable_remote_fetch` and `source_uri`."
        )


class XBRLDocumentBackend(DeclarativeDocumentBackend):
    """Backend to parse XBRL and iXBRL (inline XBRL) documents.

    XBRL (eXtensible Business Reporting Language) is a standard XML-based
    format for business and financial reporting.
    It is used globally by companies and regulators for exchanging financial
    information in a structured, machine-readable format.

    The backend accepts two input types:

    - XBRL instance documents (`.xml` / `.xbrl`) that contain only
      machine-readable data.
    - Inline XBRL (iXBRL) documents (`.htm`, `.html`, `.xhtml`) that embed
      XBRL facts in XHTML using the `ix:` namespace. This is the mandatory
      submission format for SEC EDGAR, UK Companies House, and ESEF filings.

    In both cases, the taxonomy can be supplied via `XBRLBackendOptions.taxonomy`.

    Refer to https://www.xbrl.org for more details on XBRL. In particular, refer to
    https://www.xbrl.org/Specification/taxonomy-package/REC-2016-04-19/taxonomy-package-REC-2016-04-19.html
    for details on how to provide a taxonomy package.

    This backend leverages the Arelle library for XBRL processing.
    """

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: XBRLBackendOptions | None = None,
    ) -> None:
        if options is None:
            options = XBRLBackendOptions()
        # Check if arelle is available before proceeding
        if not _XBRL_AVAILABLE:
            raise ImportError(
                "The 'arelle-release' package is required to process XBRL documents. "
                "Please install it using `pip install 'docling-slim[format-xml-xbrl]'`"
            ) from _XBRL_IMPORT_ERROR

        super().__init__(in_doc, path_or_stream)
        self.options: XBRLBackendOptions = options
        self.model_xbrl: ModelXbrl | None = None
        self._kv_idx: int = 0
        self._cells: list[GraphCell] = []
        self._links: list[GraphLink] = []
        self._hierarchy_cell_ids: dict[str, int] = {}
        self._fact_cell_ids: dict[str, list[int]] = defaultdict(list)
        self._created_links: set[tuple[int, int]] = set()

        try:
            if (
                not self.options.enable_local_fetch
                and not self.options.enable_remote_fetch
            ):
                raise OperationNotAllowed(
                    "Fetching local or remote resources is only allowed when set"
                    " explicitly. Set 'options.enable_local_fetch=True' or"
                    " 'options.enable_remote_fetch=True'. Either one or the other"
                    " needs to be enabled to load taxonomies."
                )
            # Arelle keeps the taxonomy package zip files open for the lifetime
            # of the model, which outlives this directory. On Windows, open
            # files cannot be deleted, so the cleanup of this directory may
            # leave stale files behind instead of raising.
            with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                tmp_path: Path = Path(tmpdir)
                zip_paths: list[str] = []

                if isinstance(path_or_stream, BytesIO):
                    # Preserve the original file extension so Arelle recognizes
                    # the document type (e.g. .htm for iXBRL, .xml for XBRL).
                    suffix = self.file.suffix or ".xml"
                    instance_path: Path = tmp_path / f"instance{suffix}"
                    instance_path.write_bytes(path_or_stream.getvalue())
                else:
                    instance_path = Path(shutil.copy2(path_or_stream, tmp_path))

                if self.options.taxonomy:
                    taxonomy: Path = self.options.taxonomy.resolve()
                    if not taxonomy.is_dir():
                        raise ValueError(
                            "The 'taxonomy' backend option must be a directory"
                        )
                    taxonomy_path = shutil.copytree(
                        taxonomy, tmp_path, dirs_exist_ok=True
                    )
                    zip_paths = [
                        str(item)
                        for item in taxonomy_path.iterdir()
                        if item.is_file()
                        and item.suffix.lower() == ".zip"
                        and zipfile.is_zipfile(item)
                    ]
                if zip_paths:
                    _log.debug(f"Files to be passed as taxonomy packages: {zip_paths}")
                if self.options.enable_remote_fetch and self.options.source_uri:
                    # Taxonomy files that were not provided locally are fetched
                    # from where the document was published.
                    instance_path.write_bytes(
                        _absolutize_missing_refs(
                            instance_path.read_bytes(),
                            tmp_path,
                            str(self.options.source_uri),
                        )
                    )

                # Arelle keeps the taxonomy packages and their URL remappings in
                # a process-wide registry, so a package must not outlive its load.
                with _ARELLE_LOAD_LOCK:
                    _ArellePackageManager.reset()
                    cntlr = Cntlr.Cntlr()
                    # Apply the fetch options to every document that Arelle resolves.
                    # The copy of the instance and of the taxonomy folder is always
                    # readable.
                    web_cache = FetchPolicyWebCache(
                        cntlr,
                        enable_local_fetch=self.options.enable_local_fetch,
                        enable_remote_fetch=self.options.enable_remote_fetch,
                        local_roots=[tmp_path],
                    )
                    cntlr.webCache = web_cache
                    if not self.options.enable_remote_fetch:
                        cntlr.modelManager.validateDisclosureSystem = False
                    else:
                        # TODO: parametrize the timeout?
                        web_cache.timeout = _WEB_CACHE_TIMEOUT
                        # TODO: custom set cntlr.webCache.cacheDir?
                        _log.debug(
                            f"Web Cache for remote taxonomy is: {web_cache.cacheDir}"
                        )

                    try:
                        model = cntlr.modelManager.load(
                            str(instance_path), taxonomyPackages=zip_paths
                        )
                    finally:
                        _ArellePackageManager.reset()

                if web_cache.refused:
                    raise OperationNotAllowed(
                        "XBRL resources refused by the fetch options: "
                        + "; ".join(web_cache.refused)
                    )

                if (
                    not isinstance(model, ModelXbrl)
                    or not model
                    or not model.modelDocument
                ):
                    raise ValueError("Invalid or unreadable XBRL file")
                if model.modelDocument.type not in (
                    Type.INSTANCE,
                    Type.INLINEXBRL,
                ):
                    raise ValueError(
                        "Document is not an XBRL/iXBRL instance"
                        f" (got type {Type.typeName[model.modelDocument.type]!r})"
                    )
                if model.errors:
                    raise ValueError(f"XBRL loaded with errors: {model.errors}")
                _warn_unresolved_facts(model)

            self.model_xbrl = model
            self.valid = True
        except Exception as exc:
            raise DocumentLoadError(
                "Could not initialize XBRL backend for file with hash"
                f" {self.document_hash}."
            ) from exc

    @override
    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @override
    def unload(self):
        if self.model_xbrl:
            self.model_xbrl.close()

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.XML_XBRL}

    def _get_hierarchy_cell(
        self,
        concept: ModelConcept,
    ) -> int:
        """Get existing or create new cell for a concept node."""
        qname_str = str(concept.qname)
        if qname_str not in self._hierarchy_cell_ids:
            cell_id = self._kv_idx
            self._cells.append(
                GraphCell(
                    label=GraphCellLabel.KEY,
                    cell_id=cell_id,
                    text=concept.qname.localName,
                    orig=qname_str,
                )
            )
            self._hierarchy_cell_ids[qname_str] = cell_id
            self._kv_idx += 1
        return self._hierarchy_cell_ids[qname_str]

    def _add_link(
        self,
        label: GraphLinkLabel,
        src: int,
        tgt: int,
    ) -> None:
        """Add a link if it doesn't already exist."""
        key = (src, tgt)
        if key not in self._created_links:
            self._created_links.add(key)
            self._links.append(
                GraphLink(
                    label=label,
                    source_cell_id=src,
                    target_cell_id=tgt,
                )
            )

    def _build_presentation_hierarchy(self) -> None:
        """Populate cells and links from the presentation (parent-child) linkbase.

        When an external taxonomy has not been fetched, ``fromModelObject`` on a
        relationship may be ``None``; those edges are skipped rather than raising.
        """
        assert self.model_xbrl is not None
        _log.debug("Building presentation linkbase hierarchy...")
        visited_concepts: set[str] = set()
        pre_links = self.model_xbrl.relationshipSet(
            "http://www.xbrl.org/2003/arcrole/parent-child"
        )
        for fact in self.model_xbrl.facts:
            fact_qname = str(fact.qname)
            if (
                fact.concept is None
                or not fact.concept.isNumeric
                or not fact.localName
                or not fact.value
                or fact_qname in visited_concepts
            ):
                continue

            visited_concepts.add(fact_qname)
            if fact_qname in self._fact_cell_ids:
                concept_cell_id = self._get_hierarchy_cell(fact.concept)
                for fact_cell_id in self._fact_cell_ids[fact_qname]:
                    if fact_cell_id != concept_cell_id:
                        self._add_link(
                            GraphLinkLabel.TO_CHILD,
                            concept_cell_id,
                            fact_cell_id,
                        )

            current_concept = fact.concept
            while True:
                parent = pre_links.toModelObject(current_concept)
                if not parent:
                    break
                parent_concept = parent[0].fromModelObject
                if parent_concept is None:
                    # Skip unresolved parent concepts (missing external taxonomy).
                    break
                child_cell_id = self._get_hierarchy_cell(current_concept)
                parent_cell_id = self._get_hierarchy_cell(parent_concept)
                self._add_link(
                    GraphLinkLabel.TO_CHILD,
                    parent_cell_id,
                    child_cell_id,
                )
                parent_qname = str(parent_concept.qname)
                if parent_qname in visited_concepts:
                    break
                visited_concepts.add(parent_qname)
                current_concept = parent_concept

    def _build_calculation_hierarchy(self) -> None:
        """Populate cells and links from the calculation (summation-item) linkbase."""
        assert self.model_xbrl is not None
        _log.debug("Building calculation linkbase relationships...")
        calc_links = self.model_xbrl.relationshipSet(
            "http://www.xbrl.org/2003/arcrole/summation-item"
        )
        for link in calc_links.modelRelationships:
            if link.fromModelObject is None or link.toModelObject is None:
                continue
            parent_cell_id = self._get_hierarchy_cell(link.fromModelObject)
            child_cell_id = self._get_hierarchy_cell(link.toModelObject)
            self._add_link(
                GraphLinkLabel.TO_CHILD,
                parent_cell_id,
                child_cell_id,
            )
            weight_id = self._kv_idx
            self._cells.append(
                GraphCell(
                    label=GraphCellLabel.VALUE,
                    cell_id=weight_id,
                    text=f"weight: {link.weight}",
                    orig="weight",
                )
            )
            self._kv_idx += 1
            self._add_link(
                GraphLinkLabel.TO_VALUE,
                child_cell_id,
                weight_id,
            )

    @override
    def convert(self) -> DoclingDocument:
        """Convert XBRL document to DoclingDocument using Arelle library.

        This is a placeholder implementation that creates a basic document structure.
        Full XBRL parsing using Arelle library can be implemented here.
        """
        _log.debug("Starting XBRL instance conversion...")
        if not self.is_valid() or not self.model_xbrl:
            raise RuntimeError(f"Invalid document with hash {self.document_hash}")

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/xml",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        doc_name = doc.name

        # Some metadata
        doc_type: str = ""
        doc_org: str = ""
        doc_period: str = ""
        for fact in self.model_xbrl.facts:
            if fact.qname.localName == "DocumentType" and fact.value:
                doc_type = fact.value
            if fact.qname.localName == "EntityRegistrantName" and fact.value:
                doc_org = fact.value
            if fact.qname.localName == "DocumentPeriodEndDate" and fact.value:
                doc_period = fact.value
        title = f"{doc_type} {doc_org} {doc_period}".strip()
        title = title if title else self.model_xbrl.modelDocument.basename
        doc.add_title(text=title)

        # Text blocks (as HTML)

        html_options = HTMLBackendOptions(
            enable_local_fetch=False,
            enable_remote_fetch=False,
            fetch_images=False,
            infer_furniture=False,
            add_title=False,
        )

        _log.debug("Parsing text block items and key-value items...")

        for fact in self.model_xbrl.facts:
            if fact.concept is None:
                continue
            if (
                fact.concept.type is not None
                and fact.concept.type.name == "textBlockItemType"
                and fact.value
            ):
                content = re.sub(r"\s+", " ", fact.value).strip()
                stream = BytesIO(bytes(content, encoding="utf-8"))
                in_doc = InputDocument(
                    path_or_stream=stream,
                    format=InputFormat.HTML,
                    backend=HTMLDocumentBackend,
                    backend_options=html_options,
                    filename="text_block.html",
                )
                html_backend = HTMLDocumentBackend(
                    in_doc=in_doc,
                    path_or_stream=stream,
                    options=html_options,
                )
                html_doc = html_backend.convert()
                doc = DoclingDocument.concatenate(docs=(doc, html_doc))

            if fact.concept.isNumeric and fact.localName and fact.value:
                # period
                # Arelle adjusts date-only instants and end dates by one day
                # (they denote the end of that day); `instantDate`/`endDate`
                # report the dates as declared in the instance contexts.
                # The properties are `date | None`, so guard against a
                # malformed context producing a literal "None" in the cell.
                period_text = ""
                if fact.context is not None:
                    if (
                        fact.context.isInstantPeriod
                        and fact.context.instantDate is not None
                    ):
                        period_text = str(fact.context.instantDate)
                    elif (
                        fact.context.isStartEndPeriod
                        and fact.context.startDatetime is not None
                        and fact.context.endDate is not None
                    ):
                        period_text = f"{fact.context.startDatetime.date()} - {fact.context.endDate}"

                # unit
                unit_text = ""
                if fact.unit is not None:
                    # ModelUnit.measures is always a (numerators, denominators) pair,
                    # rendered like Arelle's ModelUnit.value, e.g. "USD / shares"
                    numerators, denominators = fact.unit.measures
                    if numerators:
                        unit_text = " ".join(m.localName for m in numerators)
                        if denominators:
                            unit_text += " / " + " ".join(
                                m.localName for m in denominators
                            )

                # decimals
                decimals_text = str(fact.decimals) if fact.decimals is not None else ""

                key_id = self._kv_idx
                self._cells.append(
                    GraphCell(
                        label=GraphCellLabel.KEY,
                        cell_id=key_id,
                        text=fact.qname.localName,
                        orig=str(fact.qname),
                    )
                )
                self._fact_cell_ids[str(fact.qname)].append(key_id)
                self._kv_idx += 1

                value_cells = [
                    (f"value: {fact.value}" if fact.value else "", "value"),
                    (f"period: {period_text}" if period_text else "", "period"),
                    (f"currency: {unit_text}" if unit_text else "", "unit"),
                    (f"decimals: {decimals_text}" if decimals_text else "", "decimals"),
                ]

                for text, orig in value_cells:
                    self._cells.append(
                        GraphCell(
                            label=GraphCellLabel.VALUE,
                            cell_id=self._kv_idx,
                            text=str(text),
                            orig=str(orig),
                        )
                    )
                    self._links.append(
                        GraphLink(
                            label=GraphLinkLabel.TO_VALUE,
                            source_cell_id=key_id,
                            target_cell_id=self._kv_idx,
                        )
                    )
                    self._kv_idx += 1

        self._build_presentation_hierarchy()
        self._build_calculation_hierarchy()

        doc.name = doc_name
        if self._cells and self._links:
            graph_data: GraphData = GraphData(cells=self._cells, links=self._links)
            doc.add_key_values(graph=graph_data)

        return doc
