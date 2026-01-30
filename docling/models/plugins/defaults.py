def ocr_engines():
    from docling.models.stages.ocr.auto_ocr_model import OcrAutoModel
    from docling.models.stages.ocr.easyocr_model import EasyOcrModel
    from docling.models.stages.ocr.ocr_mac_model import OcrMacModel
    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
    from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel
    from docling.models.stages.ocr.tesseract_ocr_model import TesseractOcrModel

    return {
        "ocr_engines": [
            OcrAutoModel,
            EasyOcrModel,
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
    from docling.models.stages.picture_description.picture_description_vlm_model import (
        PictureDescriptionVlmModel,
    )

    return {
        "picture_description": [
            PictureDescriptionVlmModel,
            PictureDescriptionApiModel,
        ]
    }


def layout_engines():
    from docling.experimental.models.table_crops_layout_model import (
        TableCropsLayoutModel,
    )
    from docling.models.stages.layout.layout_model import LayoutModel

    return {
        "layout_engines": [
            LayoutModel,
            TableCropsLayoutModel,
        ]
    }


def table_structure_engines():
    from docling.models.stages.table_structure.table_structure_model import (
        TableStructureModel,
    )

    return {
        "table_structure_engines": [
            TableStructureModel,
        ]
    }
