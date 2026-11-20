"""Visitor history store: one row per client, kept in SQLite.

The daily job rewrites it from all finished visits; the API adds each visit when it finishes,
so a second visit on the same day already sees the first one. For more traffic, the same
interface can sit on Redis or a database table.
"""
import sqlite3
from pathlib import Path

import pandas as pd

from src.features import GROUPS, HISTORY_COLS

COLUMNS = ["client_id"] + HISTORY_COLS


class ClientStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        with self._connect() as con:     # a fresh deployment starts with no visitor history
            con.execute(f"CREATE TABLE IF NOT EXISTS clients ({', '.join(COLUMNS)})")
            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_client ON clients(client_id)")

    def _connect(self):
        return sqlite3.connect(self.path)

    def replace_all(self, clients: pd.DataFrame) -> None:
        frame = clients.reset_index()[COLUMNS].copy()
        frame["last_visit_dt"] = frame["last_visit_dt"].astype(str)
        with self._connect() as con:
            frame.to_sql("clients", con, if_exists="replace", index=False)
            con.execute("CREATE UNIQUE INDEX ix_client ON clients(client_id)")

    def get(self, client_ids) -> pd.DataFrame:
        ids = list(dict.fromkeys(client_ids))
        marks = ",".join("?" * len(ids))
        with self._connect() as con:
            rows = pd.read_sql(f"SELECT * FROM clients WHERE client_id IN ({marks})", con, params=ids)
        rows = rows.set_index("client_id")
        rows["last_visit_dt"] = pd.to_datetime(rows["last_visit_dt"])
        numeric = [c for c in HISTORY_COLS if c not in ("last_visit_dt", "first_medium")]
        rows[numeric] = rows[numeric].astype(float)     # an empty result would otherwise be "object"
        return rows

    def record_visit(self, visit: dict) -> None:
        """Add a finished visit: client_id, visit_dt, utm_medium, target, n_hits and event-group counts."""
        current = self.get([visit["client_id"]])
        if len(current):
            row = current.iloc[0].to_dict()
            row["prev_sessions"] += 1
            row["prev_conversions"] += visit["target"]
            row["sum_hits"] += visit["n_hits"]
            for name in GROUPS:
                row[f"past_{name}"] += visit.get(name, 0)
        else:
            row = {"prev_sessions": 1, "prev_conversions": visit["target"], "sum_hits": visit["n_hits"],
                   "first_medium": visit["utm_medium"],
                   **{f"past_{name}": visit.get(name, 0) for name in GROUPS}}
        row["last_visit_dt"] = str(pd.Timestamp(visit["visit_dt"]))
        row["last_n_hits"] = visit["n_hits"]

        values = [visit["client_id"]] + [row[c] for c in HISTORY_COLS]
        values = [v.item() if hasattr(v, "item") else v for v in values]   # numpy -> plain Python for sqlite
        with self._connect() as con:
            con.execute(f"INSERT OR REPLACE INTO clients ({','.join(COLUMNS)}) "
                        f"VALUES ({','.join('?' * len(COLUMNS))})", values)
