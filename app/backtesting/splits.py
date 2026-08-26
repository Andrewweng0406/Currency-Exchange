from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_index: list[int]
    test_index: list[int]


def yearly_expanding_splits(
    frame: pd.DataFrame,
    date_col: str = "date",
    min_train_years: int = 5,
    test_years: int = 1,
) -> list[WalkForwardSplit]:
    if frame.empty:
        return []
    dates = pd.to_datetime(frame[date_col], utc=True)
    years = sorted(dates.dt.year.unique())
    splits: list[WalkForwardSplit] = []
    for test_year in years:
        train_years = [year for year in years if year < test_year]
        if len(train_years) < min_train_years:
            continue
        test_year_group = [year for year in range(test_year, test_year + test_years)]
        train_mask = dates.dt.year.isin(train_years)
        test_mask = dates.dt.year.isin(test_year_group)
        if not test_mask.any():
            continue
        train_idx = frame.index[train_mask].tolist()
        test_idx = frame.index[test_mask].tolist()
        splits.append(
            WalkForwardSplit(
                train_start=dates.loc[train_idx].min(),
                train_end=dates.loc[train_idx].max(),
                test_start=dates.loc[test_idx].min(),
                test_end=dates.loc[test_idx].max(),
                train_index=train_idx,
                test_index=test_idx,
            )
        )
    return splits
