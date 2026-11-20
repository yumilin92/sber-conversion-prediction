"""Daily batch job: score one day's visits with the session-start blend.

    python -m src.batch_score --day 2021-12-15 --out scores/2021-12-15.parquet

Features are built from raw visits with the same code as the API (src/features.py).
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.build_store import load_history
from src.features import build_category_stats, build_client_history, history_before_each_visit, make_features
from src.scoring import score_session_start

log = logging.getLogger(__name__)


def day_features(history: pd.DataFrame, day) -> pd.DataFrame:
    """Features for one day's visits with logged events, plus their outcome when it is known."""
    day = pd.Timestamp(day).normalize()
    stats, priors = build_category_stats(history, day)
    clients = build_client_history(history[history["visit_dt"] < day])
    visits = history[history["visit_day"] == day].reset_index(drop=True)
    # earlier visits that day count as history even without logged events, as in training
    before = history_before_each_visit(visits, clients)
    logged = (visits["n_hits"] > 0).to_numpy()
    if not logged.any():
        raise ValueError(f"no visits with logged events on {day.date()}")
    features = make_features(visits[logged], stats, priors, before[logged])
    features["target"] = visits.loc[logged, "target"].to_numpy()
    features["target_in_first3"] = visits.loc[logged, "target_in_first3"].to_numpy()
    return features


def score_day(history: pd.DataFrame, day) -> pd.DataFrame:
    return score_session_start(day_features(history, day))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scores = score_day(load_history(), args.day)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.out, index=False)
    log.info("scored %d visits for %s -> %s", len(scores), args.day, args.out)


if __name__ == "__main__":
    main()
