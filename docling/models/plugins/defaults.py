def ocr_engines():
    from docling.models.stages.ocr.auto_ocr_model import OcrAutoModel
    from docling.models.stages.ocr.easyocr_model import EasyOcrModel
    from docling.models.stages.ocr.kserve_v2_ocr_model import KserveV2OcrModel
    from docling.models.stages.ocr.ocr_mac_model import OcrMacModel
    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
    from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel
    from docling.models.stages.ocr.tesseract_ocr_model import TesseractOcrModel

    return {
        "ocr_engines": [
            OcrAutoModel,
            EasyOcrModel,
            KserveV2OcrModel,
            OcrMacModel,
            RapidOcrModel,
            TesseractOcrModel,
            TesseractOcrCliModel,
        ]
    }


def picture_description():
    from docling.models.stages.picture_description.picture_description_api_model import (
        PictureDescriptionApiModel,
    )
    from docling.models.stages.picture_description.picture_description_vlm_engine_model import (
        PictureDescriptionVlmEngineModel,
    )
    from docling.models.stages.picture_description.picture_description_vlm_model import (
        PictureDescriptionVlmModel,
    )

    return {
        "picture_description": [
            PictureDescriptionVlmEngineModel,  # New engine-based (preferred)
            PictureDescriptionVlmModel,  # Legacy direct transformers
            PictureDescriptionApiModel,  # API-based
        ]
    }


def layout_engines():
    from docling.experimental.models.table_crops_layout_model import (
        TableCropsLayoutModel,
    )
    from docling.models.stages.layout.kserve_v2_layout_model import (
        KserveV2LayoutModel,
    )
    from docling.models.stages.layout.layout_model import LayoutModel
    from docling.models.stages.layout.layout_object_detection_model import (
        LayoutObjectDetectionModel,
    )

    return {
        "layout_engines": [
            KserveV2LayoutModel,
            LayoutObjectDetectionModel,
            LayoutModel,
            TableCropsLayoutModel,
        ]
    }


def table_structure_engines():
    from docling.models.stages.table_structure.kserve_v2_table_structure_model import (
        KserveV2TableStructureModel,
    )
    from docling.models.stages.table_structure.table_structure_model import (
        TableStructureModel,
    )
    from docling.models.stages.table_structure.table_structure_model_granite_vision import (
        GraniteVisionTableStructureModel,
    )
    from docling.models.stages.table_structure.table_structure_model_v2 import (
        TableStructureModelV2,
    )

    return {
        "table_structure_engines": [
            TableStructureModel,
            TableStructureModelV2,
            KserveV2TableStructureModel,
            GraniteVisionTableStructureModel,
        ]
    }
