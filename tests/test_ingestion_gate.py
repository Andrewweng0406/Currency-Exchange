from app.providers.base import ProviderResult
from scripts.ingest_phase1 import ingestion_summary


def test_supplemental_outage_keeps_core_observations_collecting():
    summary = ingestion_summary([
        ProviderResult("fx:cbc_usdtwd_close", True, rows=10),
        ProviderResult("bank:usd_sell_with_fallback", True, rows=1),
        ProviderResult("fred:batch", False, error="timeout"),
    ])
    assert summary == {"status": "DEGRADED", "missing_required": [], "failed_providers": ["fred:batch"]}


def test_failed_or_empty_core_observation_stops_pipeline():
    for results in [[], [ProviderResult("fx:cbc_usdtwd_close", True, rows=0)],
                    [ProviderResult("fx:cbc_usdtwd_close", False, error="timeout")]]:
        summary = ingestion_summary(results + [ProviderResult("bank:usd_sell_with_fallback", True, rows=1)])
        assert summary["status"] == "FAILED"
        assert "fx:cbc_usdtwd_close" in summary["missing_required"]
