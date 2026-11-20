"""Real-time API for the early-session model.

    uvicorn src.api:app

The website calls /score/early-session after a visitor's first 3 events and gets a probability
and a priority flag (top 30%: offer a chat or a callback). When a visit ends, it calls
/visits/finished, so the visitor's next visit already knows about this one.
"""
import json
import logging
import time
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.client_store import ClientStore
from src.config import STORE
from src.features import EVENT_GROUPS, TARGET_ACTIONS, clean_sessions, first_events_summary, make_features
from src.scoring import load_models, score_early_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")
app = FastAPI(title="Sber Autosubscription conversion scoring", version="2021-12.1")
MAX_BATCH = 500


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code,
             (time.perf_counter() - start) * 1000)
    return response


class Visit(BaseModel):
    session_id: str
    client_id: str
    visit_dt: datetime
    visit_number: int = Field(ge=1)
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    utm_keyword: Optional[str] = None
    device_category: Optional[str] = None
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_screen_resolution: Optional[str] = None
    device_browser: Optional[str] = None
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None
    landing_path: Optional[str] = Field(None, description="page path of the first hit, with query string")
    first_events: List[str] = Field(..., min_length=1, max_length=3, description="event_action of the first hits")


class Score(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    session_id: str
    probability: Optional[float]
    priority: bool
    model_version: str
    note: Optional[str] = None


class FinishedVisit(BaseModel):
    client_id: str
    visit_dt: datetime
    utm_medium: Optional[str] = None
    n_hits: int = Field(ge=0)
    events: List[str] = Field(default_factory=list, description="event_action of every hit in the visit")


@lru_cache
def _load_stats(mtime: float):
    stats = pd.read_parquet(STORE / "category_stats.parquet")
    priors = json.loads((STORE / "priors.json").read_text())
    return stats, priors


def load_store():
    """Category stats, reloaded automatically after the daily job rewrites them."""
    path = STORE / "category_stats.parquet"
    if not path.exists():
        raise HTTPException(status_code=503, detail="feature store not built: run python -m src.build_store")
    return _load_stats(path.stat().st_mtime)


def clients() -> ClientStore:
    return ClientStore(STORE / "clients.sqlite")


@app.get("/health")
def health():
    """The store should describe yesterday. An older store means the daily job did not run."""
    _, _, card = load_models()
    _, priors = load_store()
    built = datetime.fromtimestamp((STORE / "category_stats.parquet").stat().st_mtime)
    age_hours = (datetime.now() - built).total_seconds() / 3600
    return {"status": "ok" if age_hours < 36 else "stale",
            "model_version": card["version"], "store_day": priors["day"],
            "store_built_hours_ago": round(age_hours, 1)}


@app.post("/score/early-session", response_model=List[Score])
def score(visits: List[Visit]):
    if not visits:
        return []
    if len(visits) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"send at most {MAX_BATCH} visits per request")
    _, _, card = load_models()
    stats, priors = load_store()

    raw = pd.DataFrame([v.model_dump() for v in visits])
    early = pd.DataFrame([first_events_summary(v.first_events) for v in visits])
    frame = pd.concat([clean_sessions(raw.drop(columns="first_events")), early], axis=1)
    history = clients().get(frame["client_id"]).reindex(frame["client_id"])

    try:
        result = score_early_session(make_features(frame, stats, priors, history))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    out = []
    for v, done, (_, r) in zip(visits, early["target_in_first3"], result.iterrows(), strict=False):
        if done:   # the visitor already converted; the model was trained without such visits
            out.append(Score(session_id=v.session_id, probability=None, priority=False,
                             model_version=card["version"], note="target action in the first events"))
        else:
            out.append(Score(session_id=v.session_id, probability=float(r["probability"]),
                             priority=bool(r["priority"]), model_version=card["version"]))
    log.info("scored %d visits, %d priority", len(out), sum(s.priority for s in out))
    return out


@app.post("/visits/finished", status_code=204)
def visit_finished(visit: FinishedVisit):
    events = pd.Series(visit.events, dtype=object)
    clients().record_visit({
        "client_id": visit.client_id,
        "visit_dt": visit.visit_dt,
        "utm_medium": visit.utm_medium or "(missing)",
        "target": int(events.isin(TARGET_ACTIONS).any()),
        "n_hits": visit.n_hits,
        **{g: int(events.isin(a).sum()) for g, a in EVENT_GROUPS.items()},
    })
