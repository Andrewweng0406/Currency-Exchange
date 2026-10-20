import numpy as np
import pandas as pd

from app.models.baselines import ConstantProbabilityModel
from app.models.walkforward import walk_forward_probabilities


def test_changing_test_year_outcomes_does_not_choose_that_years_weights(monkeypatch):
    dates = pd.bdate_range('2015-01-02', '2024-12-31', tz='UTC')
    frame = pd.DataFrame({'date':dates, 'USDTWD_CLOSE':30+np.arange(len(dates))*.001})
    monkeypatch.setattr('app.models.walkforward._classification_models', lambda: {
        'coin_flip':ConstantProbabilityModel(.5), 'bull':ConstantProbabilityModel(.8),
        'bear':ConstantProbabilityModel(.2),
    })
    monkeypatch.setattr('app.models.walkforward.load_feature_frame', lambda session: frame.copy())
    original = walk_forward_probabilities(None, '5d', min_train_years=2)
    # Reverse the entire 2023+ path. Cross-year training labels are purged, and
    # current-test labels must not participate in selecting that year's weights.
    changed = frame.date.dt.year >= 2023
    frame.loc[changed,'USDTWD_CLOSE'] = 40-np.arange(changed.sum())*.01
    rebuilt = walk_forward_probabilities(None, '5d', min_train_years=2)
    old_2023 = original.loc[pd.to_datetime(original.date).dt.year==2023].reset_index(drop=True)
    new_2023 = rebuilt.loc[pd.to_datetime(rebuilt.date).dt.year==2023].reset_index(drop=True)
    assert not old_2023.target_up.equals(new_2023.target_up)
    np.testing.assert_allclose(old_2023.prob_up, new_2023.prob_up, rtol=0, atol=0)
