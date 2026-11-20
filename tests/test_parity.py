"""Training/serving parity on real data (skipped when data/processed is not available).

The model was trained on features from notebooks 03 and 06. Production builds them from raw
visits with src/features.py. If the two disagree, the model sees inputs it was never trained on
and nobody notices, so this test compares them on a full December day.

    python -m pytest tests/test_parity.py -v
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import PROCESSED

pytestmark = pytest.mark.skipif(not (PROCESSED / "features_v2.parquet").exists(),
                                reason="needs the original data in data/processed")

DAY = pd.Timestamp("2021-12-15")


@pytest.fixture(scope="module")
def history():
    from src.build_store import load_history
    return load_history()


@pytest.fixture(scope="module")
def training_features():
    ref = pd.read_parquet(PROCESSED / "features_v2.parquet")
    return ref[ref["visit_day"] == DAY].set_index("session_id")


@pytest.fixture(scope="module")
def serving_features(history):
    from src.features import build_category_stats, build_client_history, history_before_each_visit, make_features
    stats, priors = build_category_stats(history, DAY)
    clients = build_client_history(history[history["visit_dt"] < DAY])
    visits = history[history["visit_day"] == DAY].reset_index(drop=True)
    before = history_before_each_visit(visits, clients)
    logged = (visits["n_hits"] > 0).to_numpy()
    return make_features(visits[logged], stats, priors, before[logged]).set_index("session_id")


def mismatches(a: pd.Series, b: pd.Series) -> int:
    if a.dtype.kind in "fiub" or b.dtype.kind in "fiub":
        a, b = a.astype(float), b.astype(float)
        return int((~(np.isclose(a, b, rtol=1e-9) | (a.isna() & b.isna()))).sum())
    return int((a.astype(str).values != b.astype(str).values).sum())


def test_same_visits(serving_features, training_features):
    assert set(serving_features.index) == set(training_features.index)


def test_features_match_training(serving_features, training_features):
    from src.scoring import load_models
    features = load_models()[1]["features"]
    mine = serving_features.loc[training_features.index]
    bad = {c: mismatches(mine[c], training_features[c]) for c in features}
    bad = {c: n for c, n in bad.items() if n}
    # Only first_medium may differ, for clients with two visits at the same second:
    # their order was arbitrary in training and is fixed by visit_number here.
    assert set(bad) <= {"first_medium"}, bad
    assert sum(bad.values()) <= 0.001 * len(mine)


def test_scores_match_training(serving_features, training_features):
    from src.scoring import score_early_session
    ids = training_features.index[training_features["target_in_first3"] == 0]
    mine = score_early_session(serving_features.loc[ids].reset_index())["probability"]
    ref = score_early_session(training_features.loc[ids].reset_index())["probability"]
    assert np.mean(np.isclose(mine.values, ref.values, rtol=1e-6)) > 0.999


def test_api_visit_by_visit_matches_batch(history, serving_features, tmp_path, monkeypatch):
    """Replay the first 300 visits of the day through the API in time order, as the website would."""
    from src import api, build_store, config
    from src.scoring import score_early_session

    monkeypatch.setattr(config, "STORE", tmp_path)
    monkeypatch.setattr(api, "STORE", tmp_path)
    build_store.build(history, DAY, out=tmp_path)
    api._load_stats.cache_clear()
    client = TestClient(api.app)

    visits = history[history["visit_day"] == DAY].sort_values(["visit_dt", "visit_number"]).head(300)
    hits = pd.read_parquet(PROCESSED / "hits.parquet", columns=["session_id", "hit_number", "event_action"],
                           filters=[("session_id", "in", list(visits["session_id"]))])
    hits = hits.sort_values(["session_id", "hit_number"]).groupby("session_id")["event_action"].apply(list)

    raw_cols = ["session_id", "client_id", "visit_number", "utm_source", "utm_medium", "utm_campaign",
                "utm_adcontent", "utm_keyword", "device_category", "device_os", "device_brand",
                "device_screen_resolution", "device_browser", "geo_country", "geo_city"]
    compared = 0
    for _, v in visits.iterrows():
        events = hits.get(v["session_id"], [])
        if events and not v["target_in_first3"]:
            body = {**{c: (None if pd.isna(v[c]) else v[c]) for c in raw_cols},
                    "visit_dt": v["visit_dt"].isoformat(), "landing_path": v["landing_path"],
                    "first_events": events[:3]}
            resp = client.post("/score/early-session", json=[body])
            assert resp.status_code == 200, resp.text
            got = resp.json()[0]
            batch = serving_features.loc[[v["session_id"]]].reset_index()
            expected = score_early_session(batch)["probability"].iloc[0]
            assert got["probability"] == pytest.approx(expected, rel=1e-6), v["session_id"]
            compared += 1
        r = client.post("/visits/finished", json={
            "client_id": v["client_id"], "visit_dt": v["visit_dt"].isoformat(),
            "utm_medium": v["utm_medium"], "n_hits": int(v["n_hits"]), "events": list(events)})
        assert r.status_code == 204
    assert compared > 100
