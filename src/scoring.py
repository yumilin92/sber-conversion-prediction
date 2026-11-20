"""Scoring logic shared by the batch job and the API.

Two models, two use cases:
* session-start blend: daily batch scoring for marketing (the rank average needs a whole batch)
* early-session LightGBM: real-time scoring after a visitor's first 3 events (on-site actions)
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import MODELS

CATS = {"utm_medium", "device_category", "device_os", "device_brand", "device_browser",
        "geo_country", "geo_city", "first_medium", "landing_section", "landing_car_brand"}


@lru_cache
def load_models():
    import joblib
    start = joblib.load(MODELS / "session_start_blend.joblib")
    early = joblib.load(MODELS / "early_session_lgbm.joblib")
    card = json.loads((MODELS / "model_card.json").read_text())
    return start, early, card


def _categories(lgb_model, features):
    """Training categories, stored inside the LightGBM model, in feature order."""
    cat_feats = [f for f in features if f in CATS]
    stored = lgb_model.booster_.pandas_categorical
    if len(cat_feats) != len(stored):
        raise RuntimeError("model file does not match the expected categorical features")
    return dict(zip(cat_feats, stored, strict=False))


def prepare(frame: pd.DataFrame, lgb_model, features) -> pd.DataFrame:
    """Same columns and same category codes as in training; unseen categories become NaN."""
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"missing features: {sorted(missing)}")
    X = frame[features].copy()
    for col, cats in _categories(lgb_model, features).items():
        X[col] = X[col].where(X[col].isin(cats)).astype(pd.CategoricalDtype(cats))
    return X


def score_session_start(frame: pd.DataFrame) -> pd.DataFrame:
    start, _, card = load_models()
    X = prepare(frame, start["lgb"], start["features"])
    X_cb = X.assign(**{c: X[c].astype(object).fillna("(unseen)").astype(str) for c in start["cat_cols"]})
    preds = [start["lgb"].predict_proba(X)[:, 1],
             start["xgb"].predict_proba(X)[:, 1],
             start["catboost"].predict_proba(X_cb)[:, 1]]
    score = sum(rankdata(p) for p in preds) / (3 * len(X))
    # the average of ranks is not uniform, so deciles come from the rank of the blended score
    decile = np.ceil(rankdata(-score, method="ordinal") / len(X) * 10).astype(int)   # 1 = top 10%
    return pd.DataFrame({"session_id": frame["session_id"].values, "score": score, "decile": decile,
                         "model_version": card["version"]})


def score_early_session(frame: pd.DataFrame) -> pd.DataFrame:
    _, early, card = load_models()
    p = early["model"].predict_proba(prepare(frame, early["model"], early["features"]))[:, 1]
    return pd.DataFrame({"probability": p,
                         "priority": p >= card["early_session"]["priority_threshold"],
                         "model_version": card["version"]})
