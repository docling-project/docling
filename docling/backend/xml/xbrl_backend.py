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
from typing import Final, cast

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
from docling.exceptions import OperationNotAllowed

_XBRL_AVAILABLE: bool = False
_XBRL_IMPORT_ERROR: ImportError | None = None
try:
    from arelle import Cntlr  # type: ignore
    from arelle.ModelDocument import Type  # type: ignore
    from arelle.ModelDtsObject import ModelConcept  # type: ignore
    from arelle.ModelXbrl import ModelXbrl  # type: ignore

    _XBRL_AVAILABLE = True
except ImportError as e:
    _XBRL_IMPORT_ERROR = e

_log = logging.getLogger(__name__)


_WEB_CACHE_TIMEOUT: Final[int] = 10


class XBRLDocumentBackend(DeclarativeDocumentBackend):
    """Backend to parse XBRL (eXtensible Business Reporting Language) documents.

    XBRL is a standard XML-based format for business and financial reporting.
    It is used globally by companies and regulators for exchanging financial
    information in a structured, machine-readable format.

    The backend parses an XBRL instance file given a taxonomy package passed
    as a backend option.

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
                "Please install it using `pip install 'docling[xbrl]'`"
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
            with TemporaryDirectory() as tmpdir:
                tmp_path: Path = Path(tmpdir)
                zip_paths: list[str] = []
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
                        _log.debug(
                            f"Files to be passed as taxonomy packages: {zip_paths}"
                        )
                if isinstance(path_or_stream, BytesIO):
                    instance_path: Path = tmp_path / "instance.xml"
                    instance_path.write_bytes(path_or_stream.getvalue())
                elif isinstance(path_or_stream, Path):
                    instance_path = Path(shutil.copy2(path_or_stream, tmp_path))
                else:
                    raise TypeError("path_or_stream must be Path or BytesIO")

                # cntlr = Cntlr.Cntlr(logFileName="logToPrint")
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

                model = cntlr.modelManager.load(
                    str(instance_path), taxonomyPackages=zip_paths
                )
                if (
                    not isinstance(model, ModelXbrl)
                    or not model
                    or not model.modelDocument
                ):
                    raise ValueError("Invalid or unreadable XBRL file")
                if model.modelDocument.type != Type.INSTANCE:
                    raise ValueError("Document is not an XBRL instance")
                if model.errors:
                    raise ValueError(f"XBRL loaded with errors: {model.errors}")

            self.model_xbrl = model
            self.valid = True
        except Exception as exc:
            raise RuntimeError(
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

    @override
    def convert(self) -> DoclingDocument:
        """Convert XBRL document to DoclingDocument using Arelle library.

        This is a placeholder implementation that creates a basic document structure.
        Full XBRL parsing using Arelle library can be implemented here.
        """
        _log.debug("Starting XBRL instance conversion...")
        if not self.is_valid() or not self.model_xbrl:
            raise RuntimeError("Invalid document with hash {self.document_hash}")

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/xml",
            binary_hash=cast(int, self.document_hash),
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
        assert self.model_xbrl.modelDocument is not None
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
                period_text = ""
                if fact.context is not None:
                    if fact.context.isInstantPeriod:
                        period_text = str(fact.context.instantDatetime.date())
                    elif fact.context.isStartEndPeriod:
                        period_text = f"{fact.context.startDatetime.date()} - {fact.context.endDatetime.date()}"

                # unit
                unit_text = ""
                if (
                    fact.unit is not None
                    and fact.unit.measures
                    and fact.unit.measures[0]
                ):
                    unit_text = fact.unit.measures[0][0].localName

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

        # 1) presentation linkbase
        _log.debug("Building presentation linkbase hierarchy...")
        visited_concepts = set()
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

            # link fact to its concept
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

            # build concept hierarchy
            current_concept = fact.concept
            while True:
                parent = pre_links.toModelObject(current_concept)
                if not parent:
                    break
                parent_concept = parent[0].fromModelObject
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

        # 2) calculation linkbase
        _log.debug("Building calculation linkbase relationships...")
        calc_links = self.model_xbrl.relationshipSet(
            "http://www.xbrl.org/2003/arcrole/summation-item"
        )
        for link in calc_links.modelRelationships:
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

        doc.name = doc_name
        if self._cells and self._links:
            graph_data: GraphData = GraphData(cells=self._cells, links=self._links)
            doc.add_key_values(graph=graph_data)

        return doc
