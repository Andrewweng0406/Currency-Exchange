from app.models.training import _weights_from_scores
from app.models.explainability import FeatureContribution


class Score:
    def __init__(self, roc_auc, brier_score, twd_depreciation_recall):
        self.roc_auc = roc_auc
        self.brier_score = brier_score
        self.twd_depreciation_recall = twd_depreciation_recall


def test_weights_from_scores_sum_to_one():
    weights = _weights_from_scores(
        {
            "a": Score(0.6, 0.2, 0.7),
            "b": Score(0.5, 0.3, 0.5),
        }
    )
    assert round(sum(weights.values()), 8) == 1
    assert weights["a"] > weights["b"]


def test_feature_contribution_marks_non_causal():
    item = FeatureContribution("DXY_RETURN_5D", "USD_TWD_UP", 0.8, "logistic")
    assert "not causation" in item.note
