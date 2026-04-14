from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions


def test_options_kind():
    opts = GraniteVisionTableStructureOptions()
    assert opts.kind == "granite_vision_table"
