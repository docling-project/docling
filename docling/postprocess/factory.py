import logging
from functools import lru_cache

from docling.models.factories.base_factory import BaseFactory
from docling.postprocess.base_postprocessor import BasePostprocessor

logger = logging.getLogger(__name__)


class PostprocessorFactory(BaseFactory[BasePostprocessor]):
    def __init__(self, *args, **kwargs):
        super().__init__("postprocessors", *args, **kwargs)


@lru_cache
def get_postprocessor_factory(allow_external_plugins: bool = False) -> PostprocessorFactory:
    factory = PostprocessorFactory()
    factory.load_from_plugins(allow_external_plugins=allow_external_plugins)
    logger.info("Registered postprocessors: %r", factory.registered_kind)
    return factory


