import logging
from pathlib import Path
from typing import Optional

from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2
from docling.datamodel.pipeline_options import (
    LayoutOptions,
    granite_picture_description,
    smolvlm_picture_description,
)
from docling.datamodel.settings import settings
from docling.datamodel.vlm_model_specs import (
    GRANITEDOCLING_MLX,
    GRANITEDOCLING_TRANSFORMERS,
    SMOLDOCLING_MLX,
    SMOLDOCLING_TRANSFORMERS,
)
from docling.models.stages.code_formula.code_formula_model import CodeFormulaModel
from docling.models.stages.layout.layout_model import LayoutModel
from docling.models.stages.ocr.easyocr_model import EasyOcrModel
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from docling.models.stages.picture_classifier.document_picture_classifier import (
    DocumentPictureClassifier,
    DocumentPictureClassifierOptions,
)
from docling.models.stages.picture_description.picture_description_vlm_model import (
    PictureDescriptionVlmModel,
)
from docling.models.stages.table_structure.table_structure_model import (
    TableStructureModel,
)
from docling.models.utils.hf_model_download import download_hf_model

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
    with_granitedocling: bool = False,
    with_granitedocling_mlx: bool = False,
    with_smoldocling: bool = False,
    with_smoldocling_mlx: bool = False,
    with_granite_vision: bool = False,
    with_rapidocr: bool = True,
    with_easyocr: bool = False,
):
    if output_dir is None:
        output_dir = settings.cache_dir / "models"

    # Make sure the folder exists
    output_dir.mkdir(exist_ok=True, parents=True)

    if with_layout:
        _log.info("Downloading layout model...")
        LayoutModel.download_models(
            local_dir=output_dir / LayoutOptions().model_spec.model_repo_folder,
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
        pic_opts = DocumentPictureClassifierOptions()
        DocumentPictureClassifier.download_models(
            repo_id=pic_opts.repo_id,
            revision=pic_opts.revision,
            local_dir=output_dir / pic_opts.repo_cache_folder,
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
        download_hf_model(
            repo_id=smolvlm_picture_description.repo_id,
            local_dir=output_dir / smolvlm_picture_description.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_granitedocling:
        _log.info("Downloading GraniteDocling model...")
        download_hf_model(
            repo_id=GRANITEDOCLING_TRANSFORMERS.repo_id,
            local_dir=output_dir / GRANITEDOCLING_TRANSFORMERS.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_granitedocling_mlx:
        _log.info("Downloading GraniteDocling MLX model...")
        download_hf_model(
            repo_id=GRANITEDOCLING_MLX.repo_id,
            local_dir=output_dir / GRANITEDOCLING_MLX.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_smoldocling:
        _log.info("Downloading SmolDocling model...")
        download_hf_model(
            repo_id=SMOLDOCLING_TRANSFORMERS.repo_id,
            local_dir=output_dir / SMOLDOCLING_TRANSFORMERS.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_smoldocling_mlx:
        _log.info("Downloading SmolDocling MLX model...")
        download_hf_model(
            repo_id=SMOLDOCLING_MLX.repo_id,
            local_dir=output_dir / SMOLDOCLING_MLX.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_granite_vision:
        _log.info("Downloading Granite Vision model...")
        download_hf_model(
            repo_id=granite_picture_description.repo_id,
            local_dir=output_dir / granite_picture_description.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_rapidocr:
        for backend in ("torch", "onnxruntime"):
            _log.info(f"Downloading rapidocr {backend} models...")
            RapidOcrModel.download_models(
                backend=backend,
                local_dir=output_dir / RapidOcrModel._model_repo_folder,
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
