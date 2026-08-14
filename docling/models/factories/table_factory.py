from docling.models.base_table_model import BaseTableStructureModel
from docling.models.factories.base_factory import BaseFactory


class TableStructureFactory(BaseFactory[BaseTableStructureModel]):
    model_type = BaseTableStructureModel

    def __init__(self, plugin_name: str = BaseFactory.default_plugin_name) -> None:
        super().__init__("table_structure_engines", plugin_name)
