import logging

from docling.models.factories.base_factory import BaseFactory
from docling.models.picture_description_base_model import PictureDescriptionBaseModel

logger = logging.getLogger(__name__)


class PictureDescriptionFactory(BaseFactory[PictureDescriptionBaseModel]):
    def __init__(self, *args, **kwargs):
        super().__init__("picture_description", *args, **kwargs)
