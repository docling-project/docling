from api_usage.models.picture_description_api_model import (
    PictureDescriptionApiModelWithUsage,
)


def picture_description():
    return {"picture_description": [PictureDescriptionApiModelWithUsage]}
