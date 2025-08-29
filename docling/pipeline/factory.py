import logging
from functools import lru_cache

from docling.models.factories.base_factory import BaseFactory
from docling.pipeline.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


class PipelineFactory(BaseFactory[BasePipeline]):
    def __init__(self, *args, **kwargs):
        super().__init__("pipelines", *args, **kwargs)


@lru_cache
def get_pipeline_factory(allow_external_plugins: bool = False) -> PipelineFactory:
    factory = PipelineFactory()
    factory.load_from_plugins(allow_external_plugins=allow_external_plugins)
    logger.info("Registered pipelines: %r", factory.registered_kind)
    return factory


