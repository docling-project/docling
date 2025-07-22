## Enrich DoclingDocument
# This example allows to run Docling enrichment models on documents which have been already converted
# and stored as serialized DoclingDocument JSON files.

### Load modules

from pathlib import Path
from typing import Iterable, Optional

from docling_core.types.doc import BoundingBox, DocItem, DoclingDocument, NodeItem
from rich.pretty import pprint

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat, ItemAndImageEnrichmentElement
from docling.datamodel.document import InputDocument
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.models.document_picture_classifier import (
    DocumentPictureClassifier,
    DocumentPictureClassifierOptions,
)
from docling.utils.utils import chunkify

### Define batch size used for processing

BATCH_SIZE = 4

### From DocItem to the model inputs
# The following function is responsible for taking an item and applying the required pre-processing for the model.
# In this case we generate a cropped image from the document backend.


def prepare_element(
    doc: DoclingDocument,
    backend: PyPdfiumDocumentBackend,
    model: BaseItemAndImageEnrichmentModel,
    element: NodeItem,
) -> Optional[ItemAndImageEnrichmentElement]:
    if not model.is_processable(doc=doc, element=element):
        return None

    assert isinstance(element, DocItem)
    element_prov = element.prov[0]

    bbox = element_prov.bbox
    width = bbox.r - bbox.l
    height = bbox.t - bbox.b

    expanded_bbox = BoundingBox(
        l=bbox.l - width * model.expansion_factor,
        t=bbox.t + height * model.expansion_factor,
        r=bbox.r + width * model.expansion_factor,
        b=bbox.b - height * model.expansion_factor,
        coord_origin=bbox.coord_origin,
    )

    page_ix = element_prov.page_no - 1
    page_backend = backend.load_page(page_no=page_ix)
    cropped_image = page_backend.get_page_image(
        scale=model.images_scale, cropbox=expanded_bbox
    )
    return ItemAndImageEnrichmentElement(item=element, image=cropped_image)


### Iterate through the document
# This block defines the `enrich_document()` which is responsible for iterating through the document
# and batch the selected document items for running through the model.


def enrich_document(
    doc: DoclingDocument,
    backend: PyPdfiumDocumentBackend,
    model: BaseItemAndImageEnrichmentModel,
) -> DoclingDocument:
    def _prepare_elements(
        doc: DoclingDocument,
        backend: PyPdfiumDocumentBackend,
        model: BaseItemAndImageEnrichmentModel,
    ) -> Iterable[NodeItem]:
        for doc_element, _level in doc.iterate_items():
            prepared_element = prepare_element(
                doc=doc, backend=backend, model=model, element=doc_element
            )
            if prepared_element is not None:
                yield prepared_element

    for element_batch in chunkify(
        _prepare_elements(doc, backend, model),
        BATCH_SIZE,
    ):
        for element in model(doc=doc, element_batch=element_batch):  # Must exhaust!
            pass

    return doc


### Open and process
# The `main()` function which initializes the document and model objects for calling `enrich_document()`.


def main():
    data_folder = Path(__file__).parent / "../../tests/data"
    input_pdf_path = data_folder / "pdf/2206.01062.pdf"

    input_doc_path = data_folder / "groundtruth/docling_v2/2206.01062.json"

    doc = DoclingDocument.load_from_json(input_doc_path)

    in_pdf_doc = InputDocument(
        input_pdf_path,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
        filename=input_pdf_path.name,
    )
    backend = in_pdf_doc._backend

    model = DocumentPictureClassifier(
        enabled=True,
        artifacts_path=None,
        options=DocumentPictureClassifierOptions(),
        accelerator_options=AcceleratorOptions(),
    )

    doc = enrich_document(doc=doc, backend=backend, model=model)

    for pic in doc.pictures[:5]:
        print(pic.self_ref)
        pprint(pic.annotations)


if __name__ == "__main__":
    main()
