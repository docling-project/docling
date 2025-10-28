import pytest

@pytest.mark.unit
def test_parse_created_date(basic_processor):
    dp = basic_processor()
    assert dp.parse_created_date("2024-09-01") == 20240901
    assert dp.parse_created_date("2024-09") == 20240901
    assert dp.parse_created_date("2024") == 20240101
    assert dp.parse_created_date("") == 0
    assert dp.parse_created_date("invalid") == 0

@pytest.mark.unit
def test_safe_join(basic_processor):
    dp = basic_processor()
    assert dp.safe_join(["a","b"]) == "ab\n"
    assert dp.safe_join(123) == ""
