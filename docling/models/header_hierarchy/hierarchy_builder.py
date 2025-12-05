import logging

from pydantic import BaseModel

from docling.datamodel.document import ConversionResult
from docling.models.header_hierarchy.metadata_hierarchy import HierarchyBuilderMetadata
from docling.models.header_hierarchy.style_based_hierarchy import StyleBasedHierarchy
from docling.models.header_hierarchy.types.hierarchical_header import HierarchicalHeader
from docling.models.readingorder_model import ReadingOrderPageElement

_log = logging.getLogger(__name__)


class PDFHeaderHierarchyOptions(BaseModel):
    use_toc_hierarchy: bool = True
    # reset_additional_headers_to_toc: bool = True

    remove_duplicate_headers: bool = True
    infer_hierarchy_from_style: bool = True
    infer_hierarchy_from_numbering: bool = True
    min_prop_numbered: float = 0.3

    raise_on_error: bool = False


class HierarchyBuilder:
    def __init__(self, options: PDFHeaderHierarchyOptions):
        self.options = options

    def __call__(
        self,
        conv_res: ConversionResult,
        sorted_elements: list[ReadingOrderPageElement],
    ) -> HierarchicalHeader:
        root = HierarchicalHeader()
        if self.options.use_toc_hierarchy:
            try:
                hbm = HierarchyBuilderMetadata(
                    conv_res=conv_res,
                    sorted_elements=sorted_elements,
                    raise_on_error=self.options.raise_on_error,
                )
                root = hbm.infer()
            except Exception as e:
                if self.options.raise_on_error:
                    raise e
                else:
                    _log.error(
                        f"HierarchyBuilderMetadata infer failed with exception {type(e)}: '{e}'"
                    )

        if (
            self.options.infer_hierarchy_from_numbering
            or self.options.infer_hierarchy_from_style
        ) and not root.children:
            sbh = StyleBasedHierarchy(
                conv_res=conv_res,
                sorted_elements=sorted_elements,
                raise_on_error=self.options.raise_on_error,
                remove_duplicate_headers=self.options.remove_duplicate_headers,
                infer_hierarchy_from_style=self.options.infer_hierarchy_from_style,
                infer_hierarchy_from_numbering=self.options.infer_hierarchy_from_numbering,
                min_prop_numbered=self.options.min_prop_numbered,
            )
            root = sbh.process()

        return root
