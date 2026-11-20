import numpy as np
import pytest

from src.features import build_category_stats, build_client_history, history_before_each_visit, make_features
from src.scoring import score_early_session, score_session_start


@pytest.fixture(scope="module")
def features(history):
    day = history["visit_day"].max()
    stats, priors = build_category_stats(history, day)
    clients = build_client_history(history[history["visit_dt"] < day])
    visits = history[history["visit_day"] == day].reset_index(drop=True)
    return make_features(visits, stats, priors, history_before_each_visit(visits, clients))


def test_session_start_scores_are_valid_and_keep_row_order(features):
    out = score_session_start(features)
    assert (out["session_id"].values == features["session_id"].values).all()
    assert out["score"].between(0, 1).all()
    assert set(out["decile"]) <= set(range(1, 11))


def test_scoring_is_deterministic(features):
    a, b = score_early_session(features), score_early_session(features)
    assert np.allclose(a["probability"], b["probability"])


def test_missing_feature_is_rejected(features):
    with pytest.raises(ValueError, match="missing features"):
        score_early_session(features.drop(columns=["early_card_view"]))


def test_unseen_category_does_not_crash(features):
    out = score_early_session(features.assign(geo_city="Brand-new city"))
    assert out["probability"].between(0, 1).all()


def test_more_intent_means_higher_score(features):
    """Behavioural check: opening car cards early should raise the probability."""
    base = features.assign(early_card_view=0, early_catalogue=0, early_search=0)
    keen = base.assign(early_card_view=3)
    assert score_early_session(keen)["probability"].mean() > score_early_session(base)["probability"].mean()
