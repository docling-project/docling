import logging
from pathlib import Path
from typing import Optional, List, Tuple

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
from docling.models.stages.code_formula.code_formula_model import CodeFormulaModel
from docling.models.stages.layout.layout_model import LayoutModel
from docling.models.stages.ocr.easyocr_model import EasyOcrModel
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

from docling.exceptions import DoclingMultiModelDownloadError, DoclingModelDownloadError

_log = logging.getLogger(__name__)


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
    with_nemotron_ocr: bool = False,
    hf_token: Optional[str | bool] = None,
):
    if output_dir is None:
        output_dir = settings.cache_dir / "models"

    # Make sure the folder exists
    output_dir.mkdir(exist_ok=True, parents=True)

    # Track all failures
    failed_downloads: List[Tuple[str, Exception]] = []

    if with_layout:
        _log.info("Downloading layout model...")
        try:
            LayoutModel.download_models(
                local_dir=output_dir / LayoutOptions().model_spec.model_repo_folder,
                force=force,
                progress=progress,
                hf_token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"Layout model download failed: {e}")
            failed_downloads.append(("LayoutModel", e))

    if with_tableformer:
        _log.info("Downloading tableformer model...")
        try:
            TableStructureModel.download_models(
                local_dir=output_dir / TableStructureModel._model_repo_folder,
                force=force,
                progress=progress,
                hf_token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"Tableformer model download failed: {e}")
            failed_downloads.append(("TableStructureModel", e))

    if with_tableformer_v2:
        from docling.models.stages.table_structure.table_structure_model_v2 import (
            TableStructureModelV2,
        )

        _log.info("Downloading TableFormerV2 model...")
        try:
            TableStructureModelV2.download_models(
                local_dir=output_dir / TableStructureModelV2._model_repo_folder,
                force=force,
                progress=progress,
                hf_token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"TableStructureModelV2 model download failed: {e}")
            failed_downloads.append(("TableStructureModelV2", e))

    if with_picture_classifier:
        _log.info("Downloading picture classifier model...")
        pic_opts = DocumentPictureClassifierOptions.from_preset(
            "document_figure_classifier_v2"
        )
        try:
            DocumentPictureClassifier.download_models(
                repo_id=pic_opts.repo_id,
                revision=pic_opts.revision,
                local_dir=output_dir / pic_opts.repo_cache_folder,
                force=force,
                progress=progress,
                hf_token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"Picture classifier model download failed: {e}")
            failed_downloads.append(("DocumentPictureClassifier", e))

    if with_code_formula:
        _log.info("Downloading code formula model...")
        try:
            CodeFormulaModel.download_models(
                local_dir=output_dir / CodeFormulaModel._model_repo_folder,
                force=force,
                progress=progress,
                hf_token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"Code formula model download failed: {e}")
            failed_downloads.append(("CodeFormulaModel", e))

    if with_smolvlm:
        _log.info("Downloading SmolVlm model...")
        assert smolvlm_picture_description.repo_id is not None
        try:
            download_hf_model(
                repo_id=smolvlm_picture_description.repo_id,
                local_dir=output_dir / smolvlm_picture_description.repo_cache_folder,
                force=force,
                progress=progress,
                token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"SmolVlm model download failed: {e}")
            failed_downloads.append(("SmolVlm", e))

    if with_granitedocling:
        _log.info("Downloading GraniteDocling model...")
        try:
            download_hf_model(
                repo_id=GRANITEDOCLING_TRANSFORMERS.repo_id,
                local_dir=output_dir / GRANITEDOCLING_TRANSFORMERS.repo_cache_folder,
                force=force,
                progress=progress,
                token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"GraniteDocling model download failed: {e}")
            failed_downloads.append(("GraniteDocling", e))

    if with_granitedocling_mlx:
        _log.info("Downloading GraniteDocling MLX model...")
        try:
            download_hf_model(
                repo_id=GRANITEDOCLING_MLX.repo_id,
                local_dir=output_dir / GRANITEDOCLING_MLX.repo_cache_folder,
                force=force,
                progress=progress,
                token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"GraniteDocling MLX model download failed: {e}")
            failed_downloads.append(("GraniteDocling MLX", e))

    if with_granitedocling_2stage:
        _log.info("Downloading GraniteDocling 2stage model...")
        try:
            download_hf_model(
                repo_id=GRANITEDOCLING_2STAGE_TRANSFORMERS.repo_id,
                local_dir=output_dir
                / GRANITEDOCLING_2STAGE_TRANSFORMERS.repo_cache_folder,
                force=force,
                progress=progress,
                token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"GraniteDocling 2stage model download failed: {e}")
            failed_downloads.append(("GraniteDocling 2stage", e))

    if with_smoldocling:
        _log.info("Downloading SmolDocling model...")
        try:
            download_hf_model(
                repo_id=SMOLDOCLING_TRANSFORMERS.repo_id,
                local_dir=output_dir / SMOLDOCLING_TRANSFORMERS.repo_cache_folder,
                force=force,
                progress=progress,
                token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"SmolDocling model download failed: {e}")
            failed_downloads.append(("SmolDocling", e))

    if with_smoldocling_mlx:
        _log.info("Downloading SmolDocling MLX model...")
        try:
            download_hf_model(
                repo_id=SMOLDOCLING_MLX.repo_id,
                local_dir=output_dir / SMOLDOCLING_MLX.repo_cache_folder,
                force=force,
                progress=progress,
                token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"SmolDocling MLX model download failed: {e}")
            failed_downloads.append(("SmolDocling MLX", e))

    if with_granite_vision:
        _log.info("Downloading Granite Vision model...")
        assert granite_picture_description.repo_id is not None
        try:
            download_hf_model(
                repo_id=granite_picture_description.repo_id,
                local_dir=output_dir / granite_picture_description.repo_cache_folder,
                force=force,
                progress=progress,
                token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"Granite Vision model download failed: {e}")
            failed_downloads.append(("Granite Vision", e))

    if with_granite_chart_extraction:
        from docling.models.stages.chart_extraction.granite_vision import (
            ChartExtractionModelGraniteVision,
        )

        _log.info("Downloading Granite Vision Charts Extraction model...")
        try:
            ChartExtractionModelGraniteVision.download_models(
                local_dir=output_dir
                / ChartExtractionModelGraniteVision._model_repo_folder,
                force=force,
                progress=progress,
                hf_token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"Granite Vision Charts Extraction model download failed: {e}")
            failed_downloads.append(("ChartExtractionModelGraniteVision", e))

    if with_granite_chart_extraction_v4:
        from docling.models.stages.chart_extraction.granite_vision import (
            ChartExtractionModelGraniteVisionV4,
        )

        _log.info("Downloading Granite Vision 4.1 Charts Extraction model...")
        try:
            ChartExtractionModelGraniteVisionV4.download_models(
                local_dir=output_dir
                / ChartExtractionModelGraniteVisionV4._model_repo_folder,
                force=force,
                progress=progress,
                hf_token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(
                f"Granite Vision 4.1 Charts Extraction model download failed: {e}"
            )
            failed_downloads.append(("ChartExtractionModelGraniteVisionV4", e))

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
            force=force,
            progress=progress,
        )

    if with_nemotron_ocr:
        nemotron_model_dir = nemotron_ocr_model_dir()
        _log.info("Downloading nemotron-ocr-v2 model...")
        try:
            NemotronOcrModel.download_models(
                local_dir=output_dir / nemotron_model_dir,
                force=force,
                progress=progress,
                hf_token=hf_token,
            )
        except DoclingModelDownloadError as e:
            _log.error(f"Nemotron OCR model download failed: {e}")
            failed_downloads.append(("NemotronOcrModel", e))

    if failed_downloads:
        raise DoclingMultiModelDownloadError(failed_downloads)

    return output_dir
