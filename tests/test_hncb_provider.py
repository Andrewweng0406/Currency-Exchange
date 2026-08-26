import pytest

from app.providers.base import ProviderError
from app.providers.hncb import find_usd_spot_row

SAMPLE_ROWS = [
    {
        "DATE": "2026-08-26",
        "CUR_ID": "34",
        "BUY_AMT_BOARD": "31.7000",
        "SELL_AMT_BOARD": "31.8210",
        "TIME": "15:58:40",
        "DESC_CHI": "美金30天",
        "DESC_ENG": "USD 30 DAYS",
        "TYPE": "forward",
    },
    {
        "DATE": "2026-08-26",
        "CUR_ID": "2",
        "CASH_BUY_AMT_BOARD": "31.5200",
        "CASH_SELL_AMT_BOARD": "31.9900",
        "BUY_AMT_BOARD": "31.7700",
        "SELL_AMT_BOARD": "31.8700",
        "TIME": "15:58:40",
        "DESC_CHI": "美金",
        "DESC_ENG": "USD",
        "TYPE": "spot",
    },
]


def test_find_usd_spot_row_skips_forward_rows():
    row = find_usd_spot_row(SAMPLE_ROWS)
    assert row["TYPE"] == "spot"
    assert row["CUR_ID"] == "2"


def test_find_usd_spot_row_raises_when_missing():
    with pytest.raises(ProviderError):
        find_usd_spot_row([SAMPLE_ROWS[0]])
