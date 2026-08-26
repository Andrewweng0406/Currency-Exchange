from app.providers.twse import parse_number, roc_date_to_timestamp


def test_parse_number_handles_commas_and_plus_signs():
    assert parse_number("1,234") == 1234.0
    assert parse_number("+85.00") == 85.0
    assert parse_number("--") is None


def test_roc_date_to_timestamp():
    ts = roc_date_to_timestamp("115/08/25")
    assert ts.year == 2026
    assert ts.month == 8
    assert ts.day == 25
