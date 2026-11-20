"""Daily monitoring: has the traffic drifted away from what the model was trained on?

    python -m src.monitor reference            # once per model version, from November features
    python -m src.monitor report --day 2021-12-15

The report compares one day with the reference month:
* PSI of the model's probability and of every numeric feature
  (below 0.1 stable, 0.1 to 0.25 worth a look, above 0.25 investigate before trusting scores);
* share of category values the model has never seen;
* conversion rate and ROC-AUC, once the day's outcomes are known.
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import MODELS, PROCESSED
from src.scoring import CATS, load_models, prepare

log = logging.getLogger(__name__)
REFERENCE = MODELS / "monitoring_reference.json"
N_BINS = 10
WATCH, ALERT = 0.1, 0.25
TOP_VALUES = 20
# Not compared one day against a month, because a single day always looks different:
# one weekday, one value of each historical rate per category, and ages and counts that only grow.
# Historical rates are covered by the probability PSI and by the traffic mix below.
TRENDING = ("weekday", "is_weekend", "site_cr_7d")
TRENDING_SUFFIXES = ("_age_days", "_past_n", "_n_7d", "_n_30d", "_past_cr", "_cr_7d", "_cr_30d")


def monitored(features: list) -> list:
    return [c for c in features
            if c not in CATS and c not in TRENDING and not c.endswith(TRENDING_SUFFIXES)]


def psi(expected_share: np.ndarray, actual_share: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(expected_share, eps, None)
    a = np.clip(actual_share, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def bin_edges(values: pd.Series) -> list:
    edges = np.unique(np.nanquantile(values.astype(float), np.linspace(0, 1, N_BINS + 1)))
    return [-np.inf] + list(edges[1:-1]) + [np.inf]


def shares(values: pd.Series, edges: list) -> np.ndarray:
    v = values.astype(float)
    counts = pd.cut(v.dropna(), edges, include_lowest=True).value_counts(sort=False).to_numpy()
    out = np.append(counts, v.isna().sum())          # the last bin holds missing values
    return out / max(len(v), 1)


def early_probability(features: pd.DataFrame) -> np.ndarray:
    _, early, _ = load_models()
    return early["model"].predict_proba(prepare(features, early["model"], early["features"]))[:, 1]


def build_reference(features: pd.DataFrame) -> dict:
    _, early, card = load_models()
    columns = {c: features[c] for c in monitored(early["features"])}
    columns["early_session_probability"] = pd.Series(early_probability(features))
    ref = {"model_version": card["version"], "period": "2021-11", "n": len(features),
           "columns": {}, "mix": {}}
    for name, values in columns.items():
        edges = bin_edges(values)
        ref["columns"][name] = {"edges": edges, "shares": shares(values, edges).tolist()}
    for col in CATS:
        top = features[col].value_counts(normalize=True).head(TOP_VALUES)
        ref["mix"][col] = {"values": top.index.tolist(), "shares": top.tolist() + [1 - top.sum()]}
    return ref


def mix_shares(values: pd.Series, top: list) -> np.ndarray:
    counts = values.value_counts(normalize=True).reindex(top).fillna(0).to_numpy()
    return np.append(counts, 1 - counts.sum())


def report(features: pd.DataFrame, ref: dict, target: pd.Series = None) -> dict:
    values = {c: features[c] for c in ref["columns"] if c in features}
    prob = early_probability(features)
    values["early_session_probability"] = pd.Series(prob)

    drift = {name: psi(np.array(ref["columns"][name]["shares"]),
                       shares(v, ref["columns"][name]["edges"]))
             for name, v in values.items()}
    _, early, _ = load_models()
    known = dict(zip([c for c in early["features"] if c in CATS],
                     early["model"].booster_.pandas_categorical, strict=False))
    unseen = {c: float((~features[c].isin(cats)).mean()) for c, cats in known.items()}

    mix = {c: psi(np.array(m["shares"]), mix_shares(features[c], m["values"]))
           for c, m in ref["mix"].items()}

    top = lambda d: {k: round(v, 4) for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:5]}
    out = {
        "n_visits": len(features),
        "probability_psi": round(drift.pop("early_session_probability"), 4),
        "feature_psi": top(drift),
        "traffic_mix_psi": top(mix),
        "unseen_category_share": {k: round(v, 4) for k, v in unseen.items() if v > 0},
    }
    worst = max([out["probability_psi"], *out["feature_psi"].values(), *out["traffic_mix_psi"].values()])
    out["status"] = "alert" if worst > ALERT else "watch" if worst > WATCH else "ok"
    if target is not None and target.nunique() == 2:
        out["conversion_rate"] = round(float(target.mean()), 4)
        out["roc_auc"] = round(float(roc_auc_score(target, prob)), 4)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["reference", "report"])
    ap.add_argument("--day")
    args = ap.parse_args()

    if args.command == "reference":
        ref_features = pd.read_parquet(PROCESSED / "features_v2.parquet")
        ref_features = ref_features[(ref_features["visit_dt"] >= "2021-11-01")
                                    & (ref_features["visit_dt"] < "2021-12-01")
                                    & (ref_features["target_in_first3"] == 0)].reset_index(drop=True)
        REFERENCE.write_text(json.dumps(build_reference(ref_features)))
        log.info("reference saved: %d visits", len(ref_features))
        return

    if not args.day:
        ap.error("report needs --day")
    from src.batch_score import day_features
    from src.build_store import load_history
    features = day_features(load_history(), args.day)
    features = features[~features["target_in_first3"].astype(bool)].reset_index(drop=True)
    result = report(features, json.loads(REFERENCE.read_text()), features["target"])
    print(json.dumps({"day": args.day, **result}, indent=2))


if __name__ == "__main__":
    main()
