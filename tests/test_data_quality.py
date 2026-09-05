from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.schema import Base, Feature
from app.ops.data_quality import data_coverage_report, summarize_data_quality


def test_data_quality_reports_fail_without_rows():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        report = data_coverage_report(session)
    assert report["overall_status"] == "FAIL"


def test_data_quality_checks_feature_missing_ratio():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            Feature(
                observed_at_utc=datetime.now(timezone.utc),
                source="test",
                feature_set="daily_v1",
                values_json='{"USDTWD_CLOSE": 31.0, "DATA_COMPLETENESS": 1.0}',
            )
        )
        session.commit()
        report = data_coverage_report(session)
    assert any(item["feature"] == "USDTWD_CLOSE" for item in report["feature_quality"])


def test_data_quality_summary_blocks_core_latest_missing():
    report = {
        "coverage": [{"dataset": "fx_prices:USD/TWD", "status": "OK"}],
        "feature_quality": [{"feature": "USDTWD_CLOSE", "latest_missing": True, "status": "POOR"}],
    }
    summary = summarize_data_quality(report)
    assert summary.blocks_model_advice
    assert summary.status == "BLOCKING"


def test_data_quality_summary_allows_nonblocking_history_limits():
    report = {
        "coverage": [{"dataset": "fx_prices:USD/TWD", "status": "OK"}],
        "feature_quality": [{"feature": "CNH_CLOSE", "latest_missing": False, "status": "POOR"}],
    }
    summary = summarize_data_quality(report)
    assert not summary.blocks_model_advice
    assert summary.status == "LIMITED"


def test_data_quality_summary_does_not_block_on_us_yield_latest_gap():
    report = {
        "coverage": [{"dataset": "fx_prices:USD/TWD", "status": "OK"}],
        "feature_quality": [{"feature": "US2Y_CLOSE", "latest_missing": True, "status": "POOR"}],
    }
    summary = summarize_data_quality(report)
    assert not summary.blocks_model_advice
    assert summary.status == "LIMITED"
    assert "US2Y_CLOSE 最新值缺失" in summary.issues


def test_data_quality_summary_accepts_cny_proxy_for_cnh_history_gap():
    report = {
        "coverage": [{"dataset": "fx_prices:USD/TWD", "status": "OK"}],
        "feature_quality": [
            {"feature": "CNH_CLOSE", "latest_missing": False, "status": "POOR"},
            {"feature": "CNY_CLOSE", "latest_missing": False, "status": "OK"},
        ],
    }
    summary = summarize_data_quality(report)
    assert summary.status == "OK"
    assert summary.issues == []
