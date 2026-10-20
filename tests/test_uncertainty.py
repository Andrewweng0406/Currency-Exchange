import numpy as np
import pytest

from app.backtesting.uncertainty import InsufficientBlocksError, block_mean_interval
from scripts.analyze_history import block_interval


@pytest.mark.parametrize("n,block", [(4, 20), (20, 20), (39, 20), (79, 40)])
def test_short_history_cannot_create_a_zero_width_confidence_interval(n, block):
    values = np.linspace(-9., 10., n)
    with pytest.raises(InsufficientBlocksError):
        block_mean_interval(values, block)
    assert block_interval(values, block) == [None, None]


def test_two_prespecified_blocks_allow_a_reproducible_interval():
    values = np.linspace(-3., 4., 40)
    low, high = block_mean_interval(values, 20)
    assert low < values.mean() < high
    assert (low, high) == block_mean_interval(values, 20)


def test_constant_data_with_enough_blocks_has_true_zero_width():
    assert block_mean_interval(np.full(40, 7.), 20) == (7., 7.)


def test_interval_is_shift_equivariant_with_enough_blocks():
    values = np.random.default_rng(13).normal(size=200)
    original = np.asarray(block_mean_interval(values, 20))
    shifted = np.asarray(block_mean_interval(values + 100, 20))
    np.testing.assert_allclose(shifted, original + 100)


def test_missing_interval_stays_missing_through_cost_stress():
    import pandas as pd
    from scripts.review_quant_models import strategy_cost_sensitivity

    strategies = pd.DataFrame([dict(strategy='model', target_usd=10000, mean_savings_twd=.5,
                                    savings_ci_low=None, savings_ci_high=None)])
    stressed = strategy_cost_sensitivity(strategies)
    assert stressed.savings_ci_low.isna().all() and stressed.savings_ci_high.isna().all()
    assert not stressed.positive_lower_bound.any()
