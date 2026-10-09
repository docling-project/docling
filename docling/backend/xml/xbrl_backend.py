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

import logging
import re
import shutil
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final

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
from docling.datamodel.document import (
    IXBRL_MARKERS,
    XBRL_INSTANCE_NS,
    XBRL_TAXONOMY_LINKBASE_SUFFIXES,
    InputDocument,
)
from docling.exceptions import DocumentLoadError, OperationNotAllowed, SecurityError

_XBRL_AVAILABLE: bool = False
_XBRL_IMPORT_ERROR: ImportError | None = None
try:
    from arelle import (
        Cntlr,  # type: ignore
        PackageManager as _ArellePackageManager,  # type: ignore
    )
    from arelle.ModelDocument import Type  # type: ignore
    from arelle.ModelDtsObject import ModelConcept  # type: ignore
    from arelle.ModelXbrl import ModelXbrl  # type: ignore

    _XBRL_AVAILABLE = True
except ImportError as e:
    _XBRL_IMPORT_ERROR = e

_log = logging.getLogger(__name__)


_WEB_CACHE_TIMEOUT: Final[int] = 10


def _remove_transient_packages(
    pkg_mgr: Any,
    ids_before: set[str],
    cntlr: Any,
) -> None:
    """Remove filing-specific taxonomy packages added during a single conversion.

    Arelle's ``PackageManager`` is a process-level singleton. Each call to
    ``modelManager.load(..., taxonomyPackages=[...])`` registers those packages
    in the singleton. After the ``TemporaryDirectory`` that holds the extracted
    files is deleted, the registered paths become stale and corrupt the next
    conversion. This function removes only the packages whose identifiers were
    not present before this conversion started, then rebuilds the URL remapping
    table so subsequent conversions begin with a clean state.

    Standard taxonomy packages that were already registered before this
    conversion (e.g. US-GAAP or IFRS packages cached on disk) are left intact.

    Args:
        pkg_mgr: The Arelle ``PackageManager`` singleton instance.
        ids_before: Set of package identifiers that existed before this load.
        cntlr: The Arelle ``Cntlr`` used for this conversion, needed by
            ``rebuildRemappings``.
    """
    pkg_cfg = pkg_mgr.packagesConfig
    if not pkg_cfg:
        return
    packages: list[dict] = pkg_cfg.get("packages", [])
    kept = [p for p in packages if p.get("identifier") in ids_before]
    removed_count = len(packages) - len(kept)
    if removed_count:
        packages[:] = kept
        pkg_mgr.packagesConfigChanged = True
        pkg_mgr.rebuildRemappings(cntlr)
        _log.debug(
            f"Removed {removed_count} transient taxonomy package(s) from"
            " Arelle PackageManager singleton after conversion."
        )


def _find_instance_entry(zf: zipfile.ZipFile) -> str:
    """Return the ZIP entry name of the primary XBRL instance document.

    The function inspects the first few kilobytes of each candidate file in the
    archive to distinguish iXBRL (``.htm``/``.html``/``.xhtml`` with ``ix:``
    markers) from traditional XBRL instances (``.xml``/``.xbrl`` with the
    XBRL 2003 instance namespace).  Taxonomy linkbase and schema files are
    excluded from consideration.

    When multiple iXBRL documents are present (e.g. a primary report plus
    separate notes), the largest one is returned as the primary instance.

    Raises:
        ValueError: When no XBRL instance document is found in the archive.
    """
    _IXBRL_SUFFIXES = (".htm", ".html", ".xhtml")
    _INSTANCE_XML_SUFFIXES = (".xml", ".xbrl")

    ixbrl_candidates: list[tuple[int, str]] = []
    xml_candidates: list[str] = []

    for name in zf.namelist():
        name_lower = name.lower()
        if any(name_lower.endswith(suf) for suf in XBRL_TAXONOMY_LINKBASE_SUFFIXES):
            continue
        if name_lower.endswith(".xsd"):
            continue
        if any(name_lower.endswith(suf) for suf in _IXBRL_SUFFIXES):
            with zf.open(name) as f:
                head = f.read(4096)
            if any(marker in head for marker in IXBRL_MARKERS):
                ixbrl_candidates.append((zf.getinfo(name).file_size, name))
        elif any(name_lower.endswith(suf) for suf in _INSTANCE_XML_SUFFIXES):
            with zf.open(name) as f:
                head = f.read(512)
            if XBRL_INSTANCE_NS in head:
                xml_candidates.append(name)

    # Prefer iXBRL; fall back to traditional XML instance.
    if ixbrl_candidates:
        ixbrl_candidates.sort(reverse=True)
        return ixbrl_candidates[0][1]
    if xml_candidates:
        return xml_candidates[0]
    raise ValueError(
        "No XBRL instance document found in ZIP archive. "
        "Expected an iXBRL (.htm/.html) or a traditional XBRL (.xml/.xbrl) file."
    )


class XBRLDocumentBackend(DeclarativeDocumentBackend):
    """Backend to parse XBRL (eXtensible Business Reporting Language) documents.

    XBRL is a standard XML-based format for business and financial reporting.
    It is used globally by companies and regulators for exchanging financial
    information in a structured, machine-readable format.

    The backend handles two input modes:

    * **Plain instance file** (``InputFormat.XML_XBRL``): a traditional XBRL
      instance document (``.xml`` / ``.xbrl``) supplied alongside a taxonomy
      directory via :attr:`XBRLBackendOptions.taxonomy`.
    * **Self-contained ZIP** (``InputFormat.ZIP_XBRL``): a ZIP archive that
      bundles all taxonomy files together with the instance document — either a
      traditional XBRL ``.xml`` file or an inline XBRL (iXBRL) ``.htm`` file.
      This is the format distributed by SEC Edgar and other regulators.

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
                in_doc.format != InputFormat.ZIP_XBRL
                and not self.options.enable_local_fetch
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
                arelle_load_path: str
                zip_paths: list[str] = []

                if isinstance(path_or_stream, BytesIO):
                    raw_bytes: bytes = path_or_stream.getvalue()
                    is_zip = zipfile.is_zipfile(BytesIO(raw_bytes))
                elif isinstance(path_or_stream, Path):
                    raw_bytes = b""
                    is_zip = zipfile.is_zipfile(path_or_stream)
                else:
                    raise TypeError("path_or_stream must be Path or BytesIO")

                if is_zip:
                    # ZIP input: the archive is self-contained (instance + taxonomy).
                    # Write the ZIP to temp dir and ask Arelle to load via the
                    # "zip_path/entry_name" path syntax it natively understands.
                    if isinstance(path_or_stream, BytesIO):
                        zip_on_disk = tmp_path / "filing.zip"
                        zip_on_disk.write_bytes(raw_bytes)
                    else:
                        zip_on_disk = Path(shutil.copy2(path_or_stream, tmp_path))
                    pkg_dir = tmp_path / "_taxonomy_packages"
                    with zipfile.ZipFile(zip_on_disk) as zf:
                        entry_name = _find_instance_entry(zf)
                        # Extract any taxonomy package ZIPs nested inside the
                        # outer ZIP so Arelle can use them for catalog mapping.
                        pkg_members_seen = 0
                        total_pkg_bytes = 0
                        for info in zf.infolist():
                            member = info.filename
                            if (
                                not member.lower().endswith(".zip")
                                or member == entry_name
                            ):
                                continue
                            # Reject oversized members before decompression.
                            if info.file_size > options.max_file_bytes:
                                raise SecurityError(
                                    f"Taxonomy package member too large to extract: {member!r}"
                                )
                            total_pkg_bytes += info.file_size
                            if total_pkg_bytes > options.max_total_bytes:
                                raise SecurityError(
                                    "Taxonomy packages exceed total extraction size limit"
                                )
                            pkg_members_seen += 1
                            if pkg_members_seen > options.max_member_count:
                                raise SecurityError(
                                    "Too many taxonomy package members in XBRL ZIP"
                                )
                            # Flatten to bare filename and confirm the resolved
                            # destination stays inside pkg_dir (zip-slip guard).
                            pkg_dir.mkdir(exist_ok=True)
                            pkg_on_disk = (pkg_dir / Path(member).name).resolve()
                            if not pkg_on_disk.is_relative_to(pkg_dir.resolve()):
                                raise SecurityError(
                                    f"ZIP slip attempt in taxonomy package: {member!r}"
                                )
                            pkg_on_disk.write_bytes(zf.read(member))
                            zip_paths.append(str(pkg_on_disk))
                    if zip_paths:
                        _log.debug(f"Taxonomy packages extracted from ZIP: {zip_paths}")
                    arelle_load_path = str(zip_on_disk) + "/" + entry_name
                    _log.debug(f"XBRL ZIP: loading instance entry '{entry_name}'")
                else:
                    # Plain instance file: materialise to temp dir and (optionally)
                    # copy the separate taxonomy directory alongside it.
                    if isinstance(path_or_stream, BytesIO):
                        # Preserve the original file extension so Arelle recognises
                        # the document type (e.g. .htm for iXBRL, .xml for XBRL).
                        suffix = self.file.suffix or ".xml"
                        instance_path: Path = tmp_path / f"instance{suffix}"
                        instance_path.write_bytes(path_or_stream.getvalue())
                    else:
                        instance_path = Path(shutil.copy2(path_or_stream, tmp_path))

                    if options.taxonomy:
                        taxonomy: Path = options.taxonomy.resolve()
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
                            _log.debug(
                                f"Files to be passed as taxonomy packages: {zip_paths}"
                            )
                    arelle_load_path = str(instance_path)

                cntlr = Cntlr.Cntlr()
                # Disable remote access for security purposes, unless explicitly set
                if not self.options.enable_remote_fetch:
                    cntlr.webCache.workOffline = True
                    cntlr.modelManager.validateDisclosureSystem = False
                else:
                    # TODO: parametrize the timeout?
                    cntlr.webCache.timeout = _WEB_CACHE_TIMEOUT
                    # TODO: custom set cntlr.webCache.cacheDir?
                    _log.debug(
                        f"Web Cache for remote taxonomy is: {cntlr.webCache.cacheDir}"
                    )

                # Snapshot the package list so filing-specific packages added
                # during this load can be removed afterwards. This prevents
                # stale paths from a deleted TemporaryDirectory from poisoning
                # subsequent conversions in the same process.
                pkg_mgr = _ArellePackageManager.getInstance()
                pkg_cfg = pkg_mgr.packagesConfig or {}
                pkg_ids_before: set[str] = {
                    p["identifier"]
                    for p in pkg_cfg.get("packages", [])
                    if p.get("identifier")
                }

                try:
                    model = cntlr.modelManager.load(
                        arelle_load_path, taxonomyPackages=zip_paths
                    )
                finally:
                    if zip_paths:
                        _remove_transient_packages(pkg_mgr, pkg_ids_before, cntlr)

                if (
                    not isinstance(model, ModelXbrl)
                    or not model
                    or not model.modelDocument
                ):
                    raise ValueError("Invalid or unreadable XBRL file")
                if model.modelDocument.type == Type.INLINEXBRLDOCUMENTSET:
                    raise OperationNotAllowed(
                        "The XBRL ZIP contains an inline XBRL document set"
                        " (multiple instance documents).  Docling converts one"
                        " document at a time and cannot merge a multi-document"
                        " inline XBRL set.  Extract the primary instance document"
                        " and supply it directly via InputFormat.XML_XBRL."
                    )
                if model.modelDocument.type not in (
                    Type.INSTANCE,
                    Type.INLINEXBRL,
                ):
                    raise ValueError(
                        "Document is not an XBRL instance"
                        f" (got type {Type.typeName[model.modelDocument.type]!r})"
                    )
                if model.errors:
                    raise ValueError(f"XBRL loaded with errors: {model.errors}")

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
        return {InputFormat.XML_XBRL, InputFormat.ZIP_XBRL}

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
        _log.debug("Building presentation linkbase hierarchy...")
        visited_concepts: set[str] = set()
        pre_links = self.model_xbrl.relationshipSet(  # type: ignore[union-attr]
            "http://www.xbrl.org/2003/arcrole/parent-child"
        )
        for fact in self.model_xbrl.facts:  # type: ignore[union-attr]
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
                    # The parent concept could not be resolved (e.g. the
                    # external taxonomy was not fetched). Skip this edge.
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
        _log.debug("Building calculation linkbase relationships...")
        calc_links = self.model_xbrl.relationshipSet(  # type: ignore[union-attr]
            "http://www.xbrl.org/2003/arcrole/summation-item"
        )
        for link in calc_links.modelRelationships:
            if link.fromModelObject is None or link.toModelObject is None:
                # One endpoint could not be resolved; skip this relationship.
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

                # dimensions
                dimensions = []
                if fact.context is not None and fact.context.qnameDims:
                    for dim_qname, dim_value in fact.context.qnameDims.items():
                        dimensions.append(
                            (
                                f"{dim_qname.localName}: {dim_value.memberQname.localName}",
                                "dimension",
                            )
                        )

                key_id = self._kv_idx
                self._cells.append(
                    GraphCell(
                        label=GraphCellLabel.KEY,
                        cell_id=key_id,
                        text=str(fact.localName),
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
