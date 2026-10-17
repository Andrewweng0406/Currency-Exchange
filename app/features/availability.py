"""Conservative date alignment for daily feeds with documented release schedules."""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


def h10_available_dates(observation_dates: pd.Series) -> pd.Series:
    """H.10 publishes the previous week on Monday 16:15 New York time.

    Use the following Taiwan date, after the US release; roll federal holidays
    forward. Historical revisions are still not point-in-time vintages.
    """
    dates = pd.to_datetime(observation_dates, utc=True).dt.normalize()
    monday = dates - pd.to_timedelta(dates.dt.dayofweek, unit="D") + pd.Timedelta(days=7)
    business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    mapping = {value: business_day.rollforward(value) + pd.Timedelta(days=1) for value in monday.unique()}
    result = monday.map(mapping)
    # The weekly release ceased May 2006 through January 2009. Do not invent
    # historical publication times for that interval.
    result = result.mask(dates < pd.Timestamp("2009-02-01", tz="UTC"))
    return pd.to_datetime(result, utc=True)
