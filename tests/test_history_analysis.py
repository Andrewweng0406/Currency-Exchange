import numpy as np
import pandas as pd

from scripts.analyze_history import block_interval, monthly_comparison, stationary_columns


def test_block_interval_preserves_constant_and_is_reproducible():
    assert block_interval(np.full(30, 7.0)) == [7.0, 7.0]
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
