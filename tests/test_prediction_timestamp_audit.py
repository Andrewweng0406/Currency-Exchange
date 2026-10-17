from scripts.audit_recorded_predictions import audit_rows


def test_audit_distinguishes_late_weekend_and_valid_timestamps_without_certifying_skill():
    def row(key, origin, created, snapshot=None):
        return {"id": key, "model_version": "legacy", "horizon": "5d", "observed_at_utc": origin,
                "created_at_utc": created, "input_snapshot": snapshot}
    result = audit_rows([
        row(1, "2026-10-15", "2026-10-16T10:00Z"),
        row(2, "2026-10-17", "2026-10-16T23:00Z"),
        row(3, "2026-10-16", "2026-10-17T01:00Z"),
        row(4, "2026-10-16", "2026-10-16T08:00Z"),
        row(5, "2026-10-16", "2026-10-19T10:00Z"),
    ], ["2026-10-15", "2026-10-16"])
    assert [r["temporal_status"] for r in result["records"]] == [
        "created_after_next_official_close", "non_official_origin", "created_in_window_only",
        "created_before_origin_close", "created_after_conservative_deadline"]
    assert result["groups"][0]["explicit_issued_at_rows"] == 0
    assert "not proof" in result["limitation"]
