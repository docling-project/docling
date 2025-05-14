import logging
from pathlib import Path
from typing import Optional

from docling.datamodel.pipeline_options import (
    granite_picture_description,
    smoldocling_vlm_conversion_options,
    smoldocling_vlm_mlx_conversion_options,
    smolvlm_picture_description,
)
from docling.datamodel.settings import settings
from docling.models.code_formula_model import CodeFormulaModel
from docling.models.document_picture_classifier import DocumentPictureClassifier
from docling.models.easyocr_model import EasyOcrModel
from docling.models.hf_vlm_model import HuggingFaceVlmModel
from docling.models.layout_model import LayoutModel
from docling.models.picture_description_vlm_model import PictureDescriptionVlmModel
from docling.models.table_structure_model import TableStructureModel

_log = logging.getLogger(__name__)


def download_models(
    output_dir: Optional[Path] = None,
    *,
    force: bool = False,
    progress: bool = False,
    with_layout: bool = True,
    with_tableformer: bool = True,
    with_code_formula: bool = True,
    with_picture_classifier: bool = True,
    with_smolvlm: bool = False,
    with_smoldocling: bool = False,
    with_smoldocling_mlx: bool = False,
    with_granite_vision: bool = False,
    with_easyocr: bool = True,
):
    if output_dir is None:
        output_dir = settings.cache_dir / "models"

    # Make sure the folder exists
    output_dir.mkdir(exist_ok=True, parents=True)

    if with_layout:
        _log.info("Downloading layout model...")
        LayoutModel.download_models(
            local_dir=output_dir / LayoutModel._model_repo_folder,
            force=force,
            progress=progress,
        )

    if with_tableformer:
        _log.info("Downloading tableformer model...")
        TableStructureModel.download_models(
            local_dir=output_dir / TableStructureModel._model_repo_folder,
            force=force,
            progress=progress,
        )

    if with_picture_classifier:
        _log.info("Downloading picture classifier model...")
        DocumentPictureClassifier.download_models(
            local_dir=output_dir / DocumentPictureClassifier._model_repo_folder,
            force=force,
            progress=progress,
        )

    if with_code_formula:
        _log.info("Downloading code formula model...")
        CodeFormulaModel.download_models(
            local_dir=output_dir / CodeFormulaModel._model_repo_folder,
            force=force,
            progress=progress,
        )

    if with_smolvlm:
        _log.info("Downloading SmolVlm model...")
        PictureDescriptionVlmModel.download_models(
            repo_id=smolvlm_picture_description.repo_id,
            local_dir=output_dir / smolvlm_picture_description.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_smoldocling:
        _log.info("Downloading SmolDocling model...")
        HuggingFaceVlmModel.download_models(
            repo_id=smoldocling_vlm_conversion_options.repo_id,
            local_dir=output_dir / smoldocling_vlm_conversion_options.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_smoldocling_mlx:
        _log.info("Downloading SmolDocling MLX model...")
        HuggingFaceVlmModel.download_models(
            repo_id=smoldocling_vlm_mlx_conversion_options.repo_id,
            local_dir=output_dir
            / smoldocling_vlm_mlx_conversion_options.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_granite_vision:
        _log.info("Downloading Granite Vision model...")
        PictureDescriptionVlmModel.download_models(
            repo_id=granite_picture_description.repo_id,
            local_dir=output_dir / granite_picture_description.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_easyocr:
        _log.info("Downloading easyocr models...")
        EasyOcrModel.download_models(
            local_dir=output_dir / EasyOcrModel._model_repo_folder,
            force=force,
            progress=progress,
        )

    return output_dir
