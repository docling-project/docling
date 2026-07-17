from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Experimental table crops layout models
    from docling.experimental.models.table_crops_layout_model import (
        TableCropsLayoutModel,
    )

    # Layout models
    from docling.models.stages.layout.layout_model import LayoutModel
    from docling.models.stages.layout.layout_object_detection_model import (
        LayoutObjectDetectionModel,
    )

    # OCR models
    from docling.models.stages.ocr.auto_ocr_model import OcrAutoModel
    from docling.models.stages.ocr.easyocr_model import EasyOcrModel
    from docling.models.stages.ocr.kserve_v2_ocr_model import KserveV2OcrModel
    from docling.models.stages.ocr.nemotron_ocr_model import NemotronOcrModel
    from docling.models.stages.ocr.ocr_mac_model import OcrMacModel
    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
    from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel
    from docling.models.stages.ocr.tesseract_ocr_model import TesseractOcrModel
    from docling.models.stages.picture_description.picture_description_api_model import (
        PictureDescriptionApiModel,
    )

    # Picture description models
    from docling.models.stages.picture_description.picture_description_vlm_engine_model import (
        PictureDescriptionVlmEngineModel,
    )
    from docling.models.stages.picture_description.picture_description_vlm_model import (
        PictureDescriptionVlmModel,
    )

    # Table structure models
    from docling.models.stages.table_structure.table_structure_model import (
        TableStructureModel,
    )
    from docling.models.stages.table_structure.table_structure_model_granite_vision import (
        GraniteVisionTableStructureModel,
    )
    from docling.models.stages.table_structure.table_structure_model_v2 import (
        TableStructureModelV2,
    )


def ocr_engines():
    engines: list[
        type[OcrAutoModel]
        | type[EasyOcrModel]
        | type[KserveV2OcrModel]
        | type[NemotronOcrModel]
        | type[OcrMacModel]
        | type[RapidOcrModel]
        | type[TesseractOcrModel]
        | type[TesseractOcrCliModel]
    ] = []

    with suppress(ImportError):
        from docling.models.stages.ocr.auto_ocr_model import OcrAutoModel

        engines.append(OcrAutoModel)
    with suppress(ImportError):
        from docling.models.stages.ocr.easyocr_model import EasyOcrModel

        engines.append(EasyOcrModel)
    with suppress(ImportError):
        from docling.models.stages.ocr.kserve_v2_ocr_model import KserveV2OcrModel

        engines.append(KserveV2OcrModel)
    with suppress(ImportError):
        from docling.models.stages.ocr.nemotron_ocr_model import NemotronOcrModel

        engines.append(NemotronOcrModel)
    with suppress(ImportError):
        from docling.models.stages.ocr.ocr_mac_model import OcrMacModel

        engines.append(OcrMacModel)
    with suppress(ImportError):
        from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

        engines.append(RapidOcrModel)
    with suppress(ImportError):
        from docling.models.stages.ocr.tesseract_ocr_model import TesseractOcrModel

        engines.append(TesseractOcrModel)
    with suppress(ImportError):
        from docling.models.stages.ocr.tesseract_ocr_cli_model import (
            TesseractOcrCliModel,
        )

        engines.append(TesseractOcrCliModel)

    return {"ocr_engines": engines}


def picture_description():
    engines: list[
        type[PictureDescriptionVlmEngineModel]
        | type[PictureDescriptionVlmModel]
        | type[PictureDescriptionApiModel]
    ] = []

    with suppress(ImportError):
        from docling.models.stages.picture_description.picture_description_vlm_engine_model import (
            PictureDescriptionVlmEngineModel,
        )

        engines.append(PictureDescriptionVlmEngineModel)  # New engine-based (preferred)
    with suppress(ImportError):
        from docling.models.stages.picture_description.picture_description_vlm_model import (
            PictureDescriptionVlmModel,
        )

        engines.append(PictureDescriptionVlmModel)  # Legacy direct transformers
    with suppress(ImportError):
        from docling.models.stages.picture_description.picture_description_api_model import (
            PictureDescriptionApiModel,
        )

        engines.append(PictureDescriptionApiModel)  # API-based

    return {"picture_description": engines}


def layout_engines():
    engines: list[
        type[LayoutObjectDetectionModel]
        | type[LayoutModel]
        | type[TableCropsLayoutModel]
    ] = []

    with suppress(ImportError):
        from docling.models.stages.layout.layout_object_detection_model import (
            LayoutObjectDetectionModel,
        )

        engines.append(LayoutObjectDetectionModel)
    with suppress(ImportError):
        from docling.models.stages.layout.layout_model import LayoutModel

        engines.append(LayoutModel)
    with suppress(ImportError):
        from docling.experimental.models.table_crops_layout_model import (
            TableCropsLayoutModel,
        )

        engines.append(TableCropsLayoutModel)

    return {"layout_engines": engines}


def table_structure_engines():
    engines: list[
        type[TableStructureModel]
        | type[TableStructureModelV2]
        | type[GraniteVisionTableStructureModel]
    ] = []

    with suppress(ImportError):
        from docling.models.stages.table_structure.table_structure_model import (
            TableStructureModel,
        )

        engines.append(TableStructureModel)
    with suppress(ImportError):
        from docling.models.stages.table_structure.table_structure_model_v2 import (
            TableStructureModelV2,
        )

        engines.append(TableStructureModelV2)
    with suppress(ImportError):
        from docling.models.stages.table_structure.table_structure_model_granite_vision import (
            GraniteVisionTableStructureModel,
        )

        engines.append(GraniteVisionTableStructureModel)

    return {"table_structure_engines": engines}
