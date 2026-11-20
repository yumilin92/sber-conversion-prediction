"""Synthetic visits for the unit tests, so the public repo needs no real data."""
import numpy as np
import pandas as pd
import pytest

from src.features import GROUPS, clean_sessions

LANDINGS = ["sberauto.com/", "sberauto.com/cars/bmw/x5?rental_page=1", "sberauto.com/cars?city=1&rent_price=2",
            "sberauto.com/cars/skoda", "sberauto.com/promo?yclid=123", None]


def synthetic_history(n=3000, days=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2021-10-01")
    visit_dt = start + pd.to_timedelta(rng.integers(0, days * 86400, n), unit="s")
    pick = lambda values: rng.choice(np.array(values, dtype=object), n)
    raw = pd.DataFrame({
        "session_id": [f"s{i}" for i in range(n)],
        "client_id": [f"c{i}" for i in rng.integers(0, n // 3, n)],
        "visit_dt": visit_dt,
        "utm_source": pick(["src_a", "src_b", "src_c", None]),
        "utm_medium": pick(["banner", "cpc", "(none)", "cpm", "organic", "referral"]),
        "utm_campaign": pick(["camp_1", "camp_2", "camp_3", None]),
        "utm_adcontent": pick(["ad_1", "ad_2", None]),
        "utm_keyword": pick(["kw_1", None]),
        "device_category": pick(["mobile", "desktop", "tablet"]),
        "device_os": pick(["Android", "iOS", "Windows", None]),
        "device_brand": pick(["Apple", "Samsung", "Xiaomi", None]),
        "device_screen_resolution": pick(["360x720", "1920x1080", "414x896"]),
        "device_browser": pick(["Chrome", "Safari", "YaBrowser"]),
        "geo_country": "Russia",
        "geo_city": pick(["Moscow", "Saint Petersburg", "Kazan"]),
        "landing_path": pick(LANDINGS),
    })
    raw = raw.sort_values("visit_dt").reset_index(drop=True)
    raw["visit_number"] = raw.groupby("client_id").cumcount() + 1
    history = clean_sessions(raw)
    history["n_hits"] = rng.integers(0, 30, n)
    history["target"] = (rng.random(n) < 0.05).astype(int)
    for g in GROUPS:
        history[g] = rng.integers(0, 3, n)
        history[f"early_{g}"] = rng.integers(0, 2, n)
    history["target_in_first3"] = False
    no_hits = history["n_hits"] == 0
    history.loc[no_hits, ["target"] + GROUPS] = 0
    return history


@pytest.fixture(scope="session")
def history():
    return synthetic_history()


@pytest.fixture
def store(history, tmp_path, monkeypatch):
    """A feature store built from the synthetic history, used by the API."""
    from src import api, build_store, config
    day = history["visit_day"].max()
    build_store.build(history[history["visit_day"] < day], day, out=tmp_path)
    monkeypatch.setattr(config, "STORE", tmp_path)
    monkeypatch.setattr(api, "STORE", tmp_path)
    api._load_stats.cache_clear()
    return tmp_path
