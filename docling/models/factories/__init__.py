import logging
from functools import lru_cache

from docling.models.factories.ocr_factory import OcrFactory
from docling.models.factories.picture_description_factory import (
    PictureDescriptionFactory,
)

logger = logging.getLogger(__name__)


@lru_cache
def get_ocr_factory(allow_external_plugins: bool = False) -> OcrFactory:
    factory = OcrFactory()
    factory.load_from_plugins(allow_external_plugins=allow_external_plugins)
    if not factory.registered_kind:
        # 코드서빙 배포에서는 docling 이 dist 메타데이터 없이 소스로만 제공되어
        # (venv 에 docling 미설치) setuptools 엔트리포인트 스캔이 비어 OCR 엔진이
        # 0개 등록된다. 이 경우 번들된 defaults 플러그인에서 직접 시드한다.
        from docling.models.plugins import defaults

        factory.process_plugin(
            defaults.ocr_engines(), "docling_defaults", defaults.__name__
        )
    logger.info("Registered ocr engines: %r", factory.registered_kind)
    return factory


@lru_cache
def get_picture_description_factory(
    allow_external_plugins: bool = False,
) -> PictureDescriptionFactory:
    factory = PictureDescriptionFactory()
    factory.load_from_plugins(allow_external_plugins=allow_external_plugins)
    if not factory.registered_kind:
        # get_ocr_factory 와 동일한 이유의 엔트리포인트 미등록 폴백.
        from docling.models.plugins import defaults

        factory.process_plugin(
            defaults.picture_description(), "docling_defaults", defaults.__name__
        )
    logger.info("Registered picture descriptions: %r", factory.registered_kind)
    return factory
