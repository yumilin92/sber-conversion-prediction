"""Daily job: rebuild the feature stores as of the start of a day.

    python -m src.build_store --day 2021-12-16

Reads finished visits from data/processed, writes store/category_stats.parquet,
store/priors.json and store/clients.sqlite.
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.client_store import ClientStore
from src.config import PROCESSED, STORE
from src.features import GROUPS, build_category_stats, build_client_history, clean_sessions, summarise_hits

log = logging.getLogger(__name__)


def load_history() -> pd.DataFrame:
    """All visits joined with what happened in them. The hit summary is cached after the first run."""
    summary_path = PROCESSED / "visit_summary.parquet"
    if not summary_path.exists():
        log.info("summarising hits (one-off, takes a few minutes)")
        hits = pd.read_parquet(PROCESSED / "hits.parquet",
                               columns=["session_id", "hit_number", "hit_page_path", "event_action"])
        summarise_hits(hits).to_parquet(summary_path)
    summary = pd.read_parquet(summary_path)
    sessions = clean_sessions(pd.read_parquet(PROCESSED / "sessions.parquet"))
    history = sessions.join(summary, on="session_id")
    # visits without logged events: nothing happened as far as we know
    filled = ["target", "n_hits"] + GROUPS
    history[filled] = history[filled].fillna(0)
    return history


def build(history: pd.DataFrame, day, out: Path = STORE) -> None:
    day = pd.Timestamp(day).normalize()
    out.mkdir(parents=True, exist_ok=True)
    stats, priors = build_category_stats(history, day)
    stats.to_parquet(out / "category_stats.parquet", index=False)
    (out / "priors.json").write_text(json.dumps(priors, indent=2))

    clients = build_client_history(history[history["visit_dt"] < day])
    ClientStore(out / "clients.sqlite").replace_all(clients)
    log.info("store for %s: %d category rows, %d clients", day.date(), len(stats), len(clients))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="stores describe history before this day")
    args = ap.parse_args()
    build(load_history(), args.day)


if __name__ == "__main__":
    main()
