# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from docling.datamodel.asr_model_specs import (
    _DISTIL_WHISPER_OPENAI_CHECKPOINTS,
    _WHISPER_S2T_PRESETS,
    _whisper_s2t_compute_type,
)
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
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
from docling.models.stages.ocr.easyocr_model import (
    EasyOcrModel,
    _resolve_easyocr_recognition_models,
    resolve_easyocr_codes,
)
from docling.models.stages.ocr.nemotron_ocr_model import (
    NemotronOcrModel,
    nemotron_ocr_model_dir,
)
from docling.models.stages.ocr.rapid_ocr_model import (
    _RAPIDOCR_DEFAULT_LANGUAGE,
    _RAPIDOCR_MODEL_TYPE,
    RapidOcrModel,
    _parse_rapidocr_model_spec,
)
from docling.models.stages.picture_classifier.document_picture_classifier import (
    DocumentPictureClassifier,
    DocumentPictureClassifierOptions,
)
from docling.models.stages.table_structure.table_structure_model import (
    TableStructureModel,
)
from docling.models.utils.hf_model_download import download_hf_model

_log = logging.getLogger(__name__)

# Prefetched when the caller does not name specific `<backend>:<lang>`
_DEFAULT_RAPIDOCR_MODELS = (
    f"torch:{_RAPIDOCR_DEFAULT_LANGUAGE}",
    f"onnxruntime:{_RAPIDOCR_DEFAULT_LANGUAGE}",
)


def _download_whisper_models(
    model_names: list[str], output_dir: Path, *, force: bool, progress: bool
) -> None:
    """Prefetch OpenAI-format checkpoints into Whisper's download root."""
    try:
        from whisper import _MODELS  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "Native Whisper prefetch requires openai-whisper. Install docling[asr]."
        ) from exc

    unknown = set(model_names) - _MODELS.keys()
    if unknown:
        raise ValueError(f"Unknown native Whisper models: {sorted(unknown)}")

    for model_name in model_names:
        # Use the installed Whisper release's URLs and expected checksums.
        url = _MODELS[model_name]
        expected_sha256 = url.rsplit("/", 2)[-2]
        checkpoint_path = output_dir / url.rsplit("/", 1)[-1]
        if (
            checkpoint_path.is_file()
            and not force
            and _sha256_file(checkpoint_path) == expected_sha256
        ):
            continue

        _log.info("Downloading native Whisper model %s...", model_name)
        temp_file = tempfile.NamedTemporaryFile(dir=output_dir, delete=False)
        temp_path = Path(temp_file.name)
        digest = hashlib.sha256()
        try:
            with temp_file:
                with requests.get(url, stream=True, timeout=(5, 30)) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))
                    with tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        disable=not progress,
                    ) as progress_bar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                temp_file.write(chunk)
                                digest.update(chunk)
                                progress_bar.update(len(chunk))
                if digest.hexdigest() != expected_sha256:
                    raise ValueError(
                        f"SHA-256 mismatch for native Whisper model {model_name}"
                    )
            temp_path.replace(checkpoint_path)
        finally:
            temp_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_distil_whisper_models(
    model_names: list[str], output_dir: Path, *, force: bool, progress: bool
) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    unknown = set(model_names) - _DISTIL_WHISPER_OPENAI_CHECKPOINTS.keys()
    if unknown:
        raise ValueError(f"Unknown Distil-Whisper models: {sorted(unknown)}")
    if not progress:
        disable_progress_bars()

    for model_name in model_names:
        repo_id, filename, revision = _DISTIL_WHISPER_OPENAI_CHECKPOINTS[model_name]
        _log.info("Downloading Distil-Whisper model %s...", model_name)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=output_dir,
            force_download=force,
        )


def _download_whisper_s2t_models(
    model_names: list[str], output_dir: Path, *, force: bool, progress: bool
) -> None:
    try:
        import whisper_s2t  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "WhisperS2T prefetch requires whisper-s2t-reborn. Install it via "
            "`pip install 'whisper-s2t-reborn>=1.8.0'`."
        ) from exc
    from huggingface_hub.utils import disable_progress_bars

    unknown = set(model_names) - _WHISPER_S2T_PRESETS.keys()
    if unknown:
        raise ValueError(f"Unknown WhisperS2T models: {sorted(unknown)}")
    if not progress:
        disable_progress_bars()

    for model_name in model_names:
        torch_dtype = _WHISPER_S2T_PRESETS[model_name].torch_dtype
        compute_types = sorted(
            {
                _whisper_s2t_compute_type(torch_dtype, device)
                for device in ("cuda", "cpu")
            }
        )
        aligner = whisper_s2t.default_word_aligner(model_name)
        names = [model_name] if aligner is None else [model_name, aligner]
        for compute_type in compute_types:
            _log.info(
                "Downloading WhisperS2T model %s (%s)...", model_name, compute_type
            )
            for name in names:
                whisper_s2t.download_model(
                    name,
                    compute_type=compute_type,
                    cache_dir=str(output_dir),
                    force_download=force,
                )


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
    with_granite_chart_extraction_v3_3: bool = False,
    with_granite_chart_extraction_v4: bool = False,
    with_rapidocr: bool = True,
    rapidocr_models: Optional[list[str]] = None,
    rapidocr_model_size: str = _RAPIDOCR_MODEL_TYPE,
    with_easyocr: bool = False,
    easyocr_languages: Optional[list[str]] = None,  # BCP-47 tags
    with_nemotron_ocr: bool = False,
    whisper_models: Optional[list[str]] = None,
    whisper_distil_models: Optional[list[str]] = None,
    whisper_s2t_models: Optional[list[str]] = None,
):
    if easyocr_languages is not None and not with_easyocr:
        raise ValueError("easyocr_languages requires with_easyocr=True")
    if rapidocr_models is not None and not with_rapidocr:
        raise ValueError("rapidocr_models requires with_rapidocr=True")

    easyocr_recognition_models = ["english_g2", "latin_g2"]
    if easyocr_languages is not None:
        easyocr_recognition_models = _resolve_easyocr_recognition_models(
            resolve_easyocr_codes(easyocr_languages)
        )

    if output_dir is None:
        output_dir = settings.cache_dir / "models"

    # Make sure the folder exists
    output_dir.mkdir(exist_ok=True, parents=True)

    if whisper_models:
        _download_whisper_models(
            whisper_models, output_dir, force=force, progress=progress
        )

    if whisper_distil_models:
        _download_distil_whisper_models(
            whisper_distil_models, output_dir, force=force, progress=progress
        )

    if whisper_s2t_models:
        _download_whisper_s2t_models(
            whisper_s2t_models, output_dir, force=force, progress=progress
        )

    if with_layout:
        _log.info("Downloading layout model...")
        layout_spec = LayoutObjectDetectionOptions().model_spec
        # Fetch every engine variant: e.g. the ONNX engine reads from its own repo.
        layout_repos = {layout_spec.repo_id: layout_spec.revision}
        for override in layout_spec.engine_overrides.values():
            merged = override.merge_with(layout_spec.repo_id, layout_spec.revision)
            assert merged.repo_id is not None and merged.revision is not None
            layout_repos[merged.repo_id] = merged.revision
        for repo_id, revision in layout_repos.items():
            download_hf_model(
                repo_id=repo_id,
                revision=revision,
                local_dir=output_dir / repo_id.replace("/", "--"),
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

    if with_tableformer_v2:
        from docling.models.stages.table_structure.table_structure_model_v2 import (
            TableStructureModelV2,
        )

        _log.info("Downloading TableFormerV2 model...")
        TableStructureModelV2.download_models(
            local_dir=output_dir / TableStructureModelV2._model_repo_folder,
            force=force,
            progress=progress,
        )

    if with_picture_classifier:
        _log.info("Downloading picture classifier model...")
        pic_opts = DocumentPictureClassifierOptions.from_preset(
            "document_figure_classifier_v2"
        )
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
        assert smolvlm_picture_description.repo_id is not None
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

    if with_granitedocling_2stage:
        _log.info("Downloading GraniteDocling 2stage model...")
        download_hf_model(
            repo_id=GRANITEDOCLING_2STAGE_TRANSFORMERS.repo_id,
            local_dir=output_dir / GRANITEDOCLING_2STAGE_TRANSFORMERS.repo_cache_folder,
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
        assert granite_picture_description.repo_id is not None
        download_hf_model(
            repo_id=granite_picture_description.repo_id,
            local_dir=output_dir / granite_picture_description.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_granite_chart_extraction_v3_3:
        from docling.datamodel.chart_extraction_options import (
            ChartExtractionVlmEngineOptions,
        )

        preset = ChartExtractionVlmEngineOptions.get_preset("granite_vision_v3_3")
        repo_id = preset.model_spec.get_repo_id(preset.default_engine_type)
        revision = preset.model_spec.get_revision(preset.default_engine_type)
        _log.info(f"Downloading Granite Vision 3.3 Chart2CSV model ({repo_id})...")
        download_hf_model(
            repo_id=repo_id,
            revision=revision,
            local_dir=output_dir / repo_id.replace("/", "--"),
            force=force,
            progress=progress,
        )

    if with_granite_chart_extraction or with_granite_chart_extraction_v4:
        from docling.datamodel.chart_extraction_options import (
            ChartExtractionVlmEngineOptions,
        )

        preset = ChartExtractionVlmEngineOptions.get_preset("granite_vision_v4")
        repo_id = preset.model_spec.get_repo_id(preset.default_engine_type)
        revision = preset.model_spec.get_revision(preset.default_engine_type)
        _log.info(
            f"Downloading Granite Vision 4.1 Charts Extraction model ({repo_id})..."
        )
        download_hf_model(
            repo_id=repo_id,
            revision=revision,
            local_dir=output_dir / repo_id.replace("/", "--"),
            force=force,
            progress=progress,
        )

    if with_rapidocr:
        # PP-OCRv6 recognition/detection are single multilingual checkpoints, so the
        # default set already covers all ~52 v6 languages. `rapidocr_models` exists for
        # the non-v6 languages, which are served by per-script PP-OCRv4/v5 models.
        # Parsed here rather than up front: it reaches into rapidocr, an optional extra.
        for spec in (
            _parse_rapidocr_model_spec(value)
            for value in (rapidocr_models or _DEFAULT_RAPIDOCR_MODELS)
        ):
            # _parse_rapidocr_model_spec always sets user_lang.
            assert spec.user_lang is not None
            _log.info(f"Downloading rapidocr {spec.backend} {spec.user_lang} models...")
            RapidOcrModel.download_models(
                backend=spec.backend,
                lang=spec.user_lang,
                local_dir=output_dir / RapidOcrModel._model_repo_folder,
                force=force,
                progress=progress,
                model_size=rapidocr_model_size,
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
        _log.info("Downloading nemotron-ocr-v2 model...")
        NemotronOcrModel.download_models(
            local_dir=output_dir / nemotron_model_dir,
            force=force,
            progress=progress,
        )

    return output_dir
