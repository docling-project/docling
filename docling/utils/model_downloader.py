import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from docling.datamodel.pipeline_options import (
    LayoutOptions,
    granite_picture_description,
    smolvlm_picture_description,
)
from docling.datamodel.settings import settings
from docling.datamodel.vlm_model_specs import (
    GRANITEDOCLING_2STAGE_TRANSFORMERS,
    GRANITEDOCLING_MLX,
    GRANITEDOCLING_TRANSFORMERS,
    SMOLDOCLING_MLX,
    SMOLDOCLING_TRANSFORMERS,
)
from docling.exceptions import DoclingModelDownloadError, DoclingMultiModelDownloadError
from docling.models.stages.code_formula.code_formula_model import CodeFormulaModel
from docling.models.stages.layout.layout_model import LayoutModel
from docling.models.stages.ocr.easyocr_model import (
    EasyOcrModel,
    _resolve_easyocr_recognition_models,
)
from docling.models.stages.ocr.nemotron_ocr_model import (
    NemotronOcrModel,
    nemotron_ocr_model_dir,
)
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from docling.models.stages.picture_classifier.document_picture_classifier import (
    DocumentPictureClassifier,
    DocumentPictureClassifierOptions,
)
from docling.models.stages.table_structure.table_structure_model import (
    TableStructureModel,
)
from docling.models.utils.hf_model_download import download_hf_model

_log = logging.getLogger(__name__)


def _safe_download(
    model_name: str,
    download_fn: Callable[..., Any],
    failed_downloads: List[Tuple[str, Exception]],
    *args: Any,
    **kwargs: Any,
):
    """
    Executes a model download function safely, recording failures to `failed_downloads`.
    """
    _log.info(f"Downloading {model_name}...")
    try:
        download_fn(*args, **kwargs)
    except DoclingModelDownloadError as e:
        _log.error(f"{model_name} download failed: {e}")
        failed_downloads.append((model_name, e))


def download_models(
    output_dir: Optional[Path] = None,
    *,
    force: bool = False,
    progress: bool = False,
    with_layout: bool = True,
    with_tableformer: bool = True,
    with_tableformer_v2: bool = False,
    with_code_formula: bool = True,
    with_picture_classifier: bool = True,
    with_smolvlm: bool = False,
    with_granitedocling: bool = False,
    with_granitedocling_mlx: bool = False,
    with_granitedocling_2stage: bool = False,
    with_smoldocling: bool = False,
    with_smoldocling_mlx: bool = False,
    with_granite_vision: bool = False,
    with_granite_chart_extraction: bool = False,
    with_granite_chart_extraction_v4: bool = False,
    with_rapidocr: bool = True,
    with_easyocr: bool = False,
    easyocr_languages: Optional[list[str]] = None,
    with_nemotron_ocr: bool = False,
    hf_token: Optional[str | bool] = None,
):
    if easyocr_languages is not None and not with_easyocr:
        raise ValueError("easyocr_languages requires with_easyocr=True")

    easyocr_recognition_models = ["english_g2", "latin_g2"]
    if easyocr_languages is not None:
        easyocr_recognition_models = _resolve_easyocr_recognition_models(
            easyocr_languages
        )

    if output_dir is None:
        output_dir = settings.cache_dir / "models"

    # Make sure the folder exists
    output_dir.mkdir(exist_ok=True, parents=True)

    # Track all failures
    failed_downloads: List[Tuple[str, Exception]] = []

    common_kwargs = {"force": force, "progress": progress}
    hf_kwargs = {**common_kwargs, "hf_token": hf_token}
    token_kwargs = {**common_kwargs, "token": hf_token}

    if with_layout:
        _safe_download(
            "LayoutModel",
            LayoutModel.download_models,
            failed_downloads,
            local_dir=output_dir / LayoutOptions().model_spec.model_repo_folder,
            **hf_kwargs,
        )
    if with_tableformer:
        _safe_download(
            "TableStructureModel",
            TableStructureModel.download_models,
            failed_downloads,
            local_dir=output_dir / TableStructureModel._model_repo_folder,
            **hf_kwargs,
        )

    if with_tableformer_v2:
        from docling.models.stages.table_structure.table_structure_model_v2 import (
            TableStructureModelV2,
        )

        _safe_download(
            "TableFormerV2",
            TableStructureModelV2.download_models,
            failed_downloads,
            local_dir=output_dir / TableStructureModelV2._model_repo_folder,
            **hf_kwargs,
        )

    if with_picture_classifier:
        pic_opts = DocumentPictureClassifierOptions.from_preset(
            "document_figure_classifier_v2"
        )
        _safe_download(
            "DocumentPictureClassifier",
            DocumentPictureClassifier.download_models,
            failed_downloads,
            repo_id=pic_opts.repo_id,
            revision=pic_opts.revision,
            local_dir=output_dir / pic_opts.repo_cache_folder,
            **hf_kwargs,
        )

    if with_code_formula:
        _safe_download(
            "CodeFormulaModel",
            CodeFormulaModel.download_models,
            failed_downloads,
            local_dir=output_dir / CodeFormulaModel._model_repo_folder,
            **hf_kwargs,
        )

    if with_smolvlm:
        _safe_download(
            "SmolVlm",
            download_hf_model,
            failed_downloads,
            repo_id=smolvlm_picture_description.repo_id,
            local_dir=output_dir / smolvlm_picture_description.repo_cache_folder,
            **token_kwargs,
        )

    if with_granitedocling:
        _safe_download(
            "GraniteDocling",
            download_hf_model,
            failed_downloads,
            repo_id=GRANITEDOCLING_TRANSFORMERS.repo_id,
            local_dir=output_dir / GRANITEDOCLING_TRANSFORMERS.repo_cache_folder,
            **token_kwargs,
        )

    if with_granitedocling_mlx:
        _safe_download(
            "GraniteDocling MLX",
            download_hf_model,
            failed_downloads,
            repo_id=GRANITEDOCLING_MLX.repo_id,
            local_dir=output_dir / GRANITEDOCLING_MLX.repo_cache_folder,
            **token_kwargs,
        )

    if with_granitedocling_2stage:
        _safe_download(
            "GraniteDocling 2stage",
            download_hf_model,
            failed_downloads,
            repo_id=GRANITEDOCLING_2STAGE_TRANSFORMERS.repo_id,
            local_dir=output_dir / GRANITEDOCLING_2STAGE_TRANSFORMERS.repo_cache_folder,
            **token_kwargs,
        )

    if with_smoldocling:
        _safe_download(
            "SmolDocling",
            download_hf_model,
            failed_downloads,
            repo_id=SMOLDOCLING_TRANSFORMERS.repo_id,
            local_dir=output_dir / SMOLDOCLING_TRANSFORMERS.repo_cache_folder,
            **token_kwargs,
        )

    if with_smoldocling_mlx:
        _safe_download(
            "SmolDocling MLX",
            download_hf_model,
            failed_downloads,
            repo_id=SMOLDOCLING_MLX.repo_id,
            local_dir=output_dir / SMOLDOCLING_MLX.repo_cache_folder,
            **token_kwargs,
        )

    if with_granite_vision:
        assert granite_picture_description.repo_id is not None
        _safe_download(
            "Granite Vision",
            download_hf_model,
            failed_downloads,
            repo_id=granite_picture_description.repo_id,
            local_dir=output_dir / granite_picture_description.repo_cache_folder,
            **token_kwargs,
        )

    if with_granite_chart_extraction:
        from docling.models.stages.chart_extraction.granite_vision import (
            ChartExtractionModelGraniteVision,
        )

        _safe_download(
            "ChartExtractionModelGraniteVision",
            ChartExtractionModelGraniteVision.download_models,
            failed_downloads,
            local_dir=output_dir / ChartExtractionModelGraniteVision._model_repo_folder,
            **hf_kwargs,
        )

    if with_granite_chart_extraction_v4:
        from docling.models.stages.chart_extraction.granite_vision import (
            ChartExtractionModelGraniteVisionV4,
        )

        _safe_download(
            "ChartExtractionModelGraniteVisionV4",
            ChartExtractionModelGraniteVisionV4.download_models,
            failed_downloads,
            local_dir=output_dir
            / ChartExtractionModelGraniteVisionV4._model_repo_folder,
            **hf_kwargs,
        )

    if with_rapidocr:
        for backend in ("torch", "onnxruntime"):
            for lang in ("chinese", "english"):
                _log.info(f"Downloading rapidocr {backend} {lang} models...")
                RapidOcrModel.download_models(
                    backend=backend,
                    local_dir=output_dir / RapidOcrModel._model_repo_folder,
                    force=force,
                    progress=progress,
                    lang=lang,
                )

    if with_easyocr:
        _log.info("Downloading easyocr models...")
        EasyOcrModel.download_models(
            local_dir=output_dir / EasyOcrModel._model_repo_folder,
            recognition_models=easyocr_recognition_models,
            force=force,
            progress=progress,
        )

    if with_nemotron_ocr:
        nemotron_model_dir = nemotron_ocr_model_dir()
        _safe_download(
            "NemotronOcrModel",
            NemotronOcrModel.download_models,
            failed_downloads,
            local_dir=output_dir / nemotron_model_dir,
            **hf_kwargs,
        )

    if failed_downloads:
        raise DoclingMultiModelDownloadError(failed_downloads)

    return output_dir
