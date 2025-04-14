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
    logger.info("Registered ocr engines: %r", factory.registered_kind)
    return factory


@lru_cache
def get_picture_description_factory(
    allow_external_plugins: bool = False,
) -> PictureDescriptionFactory:
    factory = PictureDescriptionFactory()
    factory.load_from_plugins(allow_external_plugins=allow_external_plugins)
    logger.info("Registered picture descriptions: %r", factory.registered_kind)
    return factory
