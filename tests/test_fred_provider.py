from io import BytesIO
from zipfile import ZipFile

from app.providers.fred import FredCsvProvider


class FakeResponse:
    text = "observation_date,DGS2,DGS10\n2026-09-03,4.34,4.77\n2026-09-04,.,4.78\n"
    content = b""


class FakeZipResponse:
    text = ""
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("daily.csv", "observation_date,DGS2,DGS10\n2026-09-03,4.34,4.77\n")
    content = buffer.getvalue()


def test_fetch_series_batch_splits_series(monkeypatch):
    provider = FredCsvProvider()
    monkeypatch.setattr(provider, "get", lambda *args, **kwargs: FakeResponse())

    frames = provider.fetch_series_batch({"us_2y": "DGS2", "us_10y": "DGS10"})

    assert set(frames) == {"us_2y", "us_10y"}
    assert len(frames["us_2y"]) == 1
    assert len(frames["us_10y"]) == 2
    assert frames["us_2y"].iloc[0]["value"] == 4.34


def test_fetch_series_batch_supports_fred_zip_response(monkeypatch):
    provider = FredCsvProvider()
    monkeypatch.setattr(provider, "get", lambda *args, **kwargs: FakeZipResponse())

    frames = provider.fetch_series_batch({"us_2y": "DGS2", "us_10y": "DGS10"})

    assert frames["us_2y"].iloc[0]["value"] == 4.34
    assert frames["us_10y"].iloc[0]["value"] == 4.77
