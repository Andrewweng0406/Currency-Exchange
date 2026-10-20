import numpy as np
import pandas as pd

from scripts.analyze_history import block_interval, monthly_comparison, stationary_columns


def test_inventory_covers_forward_evidence_and_tables_without_observation_time():
    from sqlalchemy import Column, Integer, MetaData, Table, create_engine
    from sqlalchemy.orm import Session
    from app.database.schema import Base, ForwardOutcome, ForwardSnapshot
    from scripts.analyze_history import inventory

    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    # Reflection must also quote identifiers and tolerate tables without a time column.
    extra = Table('external " counts', MetaData(), Column('id', Integer, primary_key=True))
    extra.create(engine)
    origin = pd.Timestamp('2026-10-16', tz='UTC').to_pydatetime()
    target = pd.Timestamp('2026-10-19', tz='UTC').to_pydatetime()
    with Session(engine) as session:
        snapshot = ForwardSnapshot(protocol_hash='a' * 64, protocol_json='{}', origin_utc=origin,
                                   horizon='1d', prediction_id=1, issued_at_utc=origin,
                                   recorded_at_utc=origin, payload_json='{}', payload_hash='b' * 64)
        session.add(snapshot)
        session.flush()
        session.add(ForwardOutcome(snapshot_id=snapshot.id, target_utc=target, evaluated_at_utc=target,
                                  close=32., realized_return=.01, target_up=1, evidence_json='{}'))
        session.execute(extra.insert().values(id=1))
        session.commit()
        rows, _ = inventory(session)
    tables = {row['table']: row for row in rows}
    assert set(Base.metadata.tables).issubset(tables)
    assert tables['forward_snapshots']['time_column'] == 'origin_utc'
    assert tables['forward_snapshots']['rows'] == 1
    assert tables['forward_snapshots']['start'].date() == origin.date()
    assert tables['forward_outcomes']['time_column'] == 'target_utc'
    assert tables['forward_outcomes']['end'].date() == target.date()
    assert tables[extra.name] == dict(table=extra.name, rows=1, time_column=None, start=None, end=None)


def test_block_interval_preserves_constant_and_is_reproducible():
    assert block_interval(np.full(60, 7.0)) == [7.0, 7.0]
    values = np.arange(40, dtype=float)
    first = block_interval(values, block_size=5)
    assert first == block_interval(values, block_size=5)
    assert first[0] < values.mean() < first[1]


def test_stationary_research_does_not_include_price_levels_or_targets():
    columns = ['USDTWD_CLOSE', 'USDTWD_SMA_20D', 'DXY_CLOSE', 'USDTWD_RETURN_5D', 'US2Y_CHANGE_5D']
    assert stationary_columns(columns) == ['USDTWD_RETURN_5D', 'US2Y_CHANGE_5D']


def test_monthly_research_excludes_partial_month_and_executes_after_signal():
    dates = pd.bdate_range('2024-01-01', '2024-02-15', tz='UTC')
    frame = pd.DataFrame({'date': dates, 'USDTWD_CLOSE': 30 + np.arange(len(dates)) * .1})
    predictions = pd.DataFrame({'date': dates, 'horizon': 5, 'existing_ensemble': .8})
    monthly, summary = monthly_comparison(frame, predictions)
    assert monthly.month.tolist() == ['2024-01']
    assert monthly.iloc[0].model_execution_date == '2024-01-02'
    assert monthly.iloc[0].model_rate == 30.1
    assert monthly.iloc[0].fixed_rate == 30
    model = summary[summary.strategy == 'model'].iloc[0]
    assert np.isclose(model.mean_savings_twd, -1000)


def test_fixed_tranches_follow_calendar_dates_without_future_prices():
    dates = pd.bdate_range('2024-01-01', '2024-02-15', tz='UTC')
    frame = pd.DataFrame({'date': dates, 'USDTWD_CLOSE': 30 + np.arange(len(dates)) * .1})
    predictions = pd.DataFrame({'date': dates, 'horizon': 5, 'existing_ensemble': .5})
    monthly, _ = monthly_comparison(frame, predictions)
    # Jan 1, 8, 15 and 22 are Mondays: positions 0, 5, 10, 15.
    assert np.isclose(monthly.iloc[0].tranche_rate, np.mean([30, 30.5, 31, 31.5]))


def test_cost_stress_compares_equal_budgets_and_only_incremental_costs():
    from scripts.review_quant_models import strategy_cost_sensitivity

    summaries = pd.DataFrame([
        dict(strategy=name, target_usd=10000, mean_savings_twd=mean,
             savings_ci_low=low, savings_ci_high=high)
        for name, mean, low, high in [('fixed', 0, 0, 0), ('model', 268, -877, 1401),
                                     ('tranche', -8, -639, 710)]
    ])
    results = strategy_cost_sensitivity(summaries)
    # Identical single-order fees cancel; a .03 adverse premium costs 300 TWD.
    model = results[(results.strategy == 'model') &
                    (results.assumed_extra_premium_twd_per_usd == .03)]
    assert model.mean_savings_after_cost_twd.eq(-32).all()
    assert model.savings_ci_low.eq(-1177).all()
    assert model.savings_ci_high.eq(1101).all()
    tranche = results[(results.strategy == 'tranche') &
                      (results.assumed_extra_premium_twd_per_usd == .03) &
                      (results.fee_twd_per_order == 100)].iloc[0]
    assert tranche.extra_cost_twd == 600  # 10,000 USD once, plus three extra orders.
    assert tranche.mean_savings_after_cost_twd == -608
    assert results[results.strategy == 'fixed'].extra_cost_twd.eq(0).all()
    assert not results.positive_lower_bound.any()
