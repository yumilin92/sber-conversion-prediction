import pandas as pd

from src.features import (
    build_category_stats,
    build_client_history,
    first_events_summary,
    history_before_each_visit,
    make_features,
)


def features_for_day(history, day):
    stats, priors = build_category_stats(history, day)
    clients = build_client_history(history[history["visit_dt"] < day])
    visits = history[history["visit_day"] == day].reset_index(drop=True)
    return make_features(visits, stats, priors, history_before_each_visit(visits, clients))


def test_future_visits_do_not_change_features(history):
    """No leakage: anything after the day being scored must not affect its features."""
    day = pd.Timestamp("2021-10-20")
    full = features_for_day(history, day)
    cut = features_for_day(history[history["visit_day"] <= day], day)
    pd.testing.assert_frame_equal(full, cut)


def test_todays_outcomes_do_not_change_category_stats(history):
    day = pd.Timestamp("2021-10-20")
    flipped = history.copy()
    today = flipped["visit_day"] == day
    flipped.loc[today, "target"] = 1 - flipped.loc[today, "target"]
    pd.testing.assert_frame_equal(build_category_stats(history, day)[0], build_category_stats(flipped, day)[0])


def test_new_visitor_gets_empty_history(history):
    day = history["visit_day"].max()
    stats, priors = build_category_stats(history, day)
    visit = history[history["visit_day"] == day].head(1).assign(client_id="never-seen")
    empty = build_client_history(history.iloc[0:0])
    f = make_features(visit, stats, priors, history_before_each_visit(visit, empty))
    assert f.loc[0, "prev_sessions"] == 0
    assert pd.isna(f.loc[0, "days_since_prev"])
    assert f.loc[0, "first_medium"] == visit["utm_medium"].iloc[0]


def test_earlier_visit_same_day_counts_as_history(history):
    day = history["visit_day"].max()
    visits = history[history["visit_day"] == day].head(2).assign(client_id="same-person")
    visits["visit_dt"] = [day + pd.Timedelta(hours=9), day + pd.Timedelta(hours=10)]
    before = history_before_each_visit(visits, build_client_history(history.iloc[0:0]))
    assert before["prev_sessions"].tolist() == [0, 1]
    assert before["last_n_hits"].iloc[1] == visits["n_hits"].iloc[0]


def test_unseen_campaign_gets_prior_rate_and_age_zero(history):
    day = history["visit_day"].max()
    stats, priors = build_category_stats(history, day)
    visit = history[history["visit_day"] == day].head(1).assign(utm_campaign="brand-new")
    clients = build_client_history(history[history["visit_dt"] < day])
    f = make_features(visit, stats, priors, history_before_each_visit(visit, clients))
    assert f.loc[0, "utm_campaign_past_cr"] == priors["past"]
    assert f.loc[0, "utm_campaign_past_n"] == 0
    assert f.loc[0, "utm_campaign_age_days"] == 0


def test_landing_page_parsing(history):
    day = history["visit_day"].max()
    stats, priors = build_category_stats(history, day)
    visits = history[history["visit_day"] == day].head(3).reset_index(drop=True)
    visits["landing_path"] = ["sberauto.com/cars/bmw/x5?rental_page=1&city=1", "sberauto.com/", None]
    clients = build_client_history(history[history["visit_dt"] < day])
    f = make_features(visits, stats, priors, history_before_each_visit(visits, clients))
    assert f["landing_section"].tolist() == ["cars", "home", "(missing)"]
    assert f["landing_car_brand"].tolist() == ["bmw", "(none)", "(none)"]
    assert f["lp_rental_page"].tolist() == [1, 0, 0]
    assert f["lp_city_filter"].tolist() == [1, 0, 0]


def test_first_events_summary():
    out = first_events_summary(["view_card", "photos", "sub_car_claim_click", "view_card"])
    assert out["early_card_view"] == 1          # only the first 3 events count
    assert out["early_photos"] == 1
    assert out["target_in_first3"] is True
