def ocr_engines():
    from docling.models.auto_ocr_model import OcrAutoModel
    from docling.models.deepseek_ocr_model import DeepSeekOcrModel
    from docling.models.easyocr_model import EasyOcrModel
    from docling.models.ocr_mac_model import OcrMacModel
    from docling.models.rapid_ocr_model import RapidOcrModel
    from docling.models.tesseract_ocr_cli_model import TesseractOcrCliModel
    from docling.models.tesseract_ocr_model import TesseractOcrModel

    return {
        "ocr_engines": [
            OcrAutoModel,
            DeepSeekOcrModel,
            EasyOcrModel,
            OcrMacModel,
            RapidOcrModel,
            TesseractOcrModel,
            TesseractOcrCliModel,
        ]
    }


def picture_description():
    from docling.models.picture_description_api_model import PictureDescriptionApiModel
    from docling.models.picture_description_vlm_model import PictureDescriptionVlmModel

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
    from docling.models.layout_model import LayoutModel

    return {
        "layout_engines": [
            LayoutModel,
            TableCropsLayoutModel,
        ]
    }


def table_structure_engines():
    from docling.models.table_structure_model import TableStructureModel

    return {
        "table_structure_engines": [
            TableStructureModel,
        ]
    }
