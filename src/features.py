"""Features for serving, built from raw visits.

In training (notebooks 03 and 06) the features were computed over the whole history at once.
In production a visit arrives alone, so the history lives in two stores:

* category stats: past and recent conversion rate of every traffic source, campaign, city, ...
  Rebuilt once a day from all finished visits before that day.
* visitor history: one row per client, updated each time a visit finishes.

tests/test_parity.py checks that this module reproduces the training features exactly.
"""
import numpy as np
import pandas as pd

TARGET_ACTIONS = [
    "sub_car_claim_click", "sub_car_claim_submit_click", "sub_open_dialog_click",
    "sub_custom_question_submit_click", "sub_call_number_click",
    "sub_callback_submit_click", "sub_submit_success", "sub_car_request_submit_click",
]
EVENT_GROUPS = {
    "card_view": ["view_card", "view_new_card", "go_to_car_card", "sub_car_page"],
    "search": ["search_form_region", "search_form_search_btn", "search_form_mark_select",
               "search_form_model_select", "search_form_search_car_type_select",
               "sap_search_form_cost_to", "sap_search_form_cost_from"],
    "photos": ["photos", "photos_all"],
    "catalogue": ["sub_view_cars_click", "pagination_click", "view_more_click", "listing_ads_sort"],
    "quiz": ["quiz_show"],
    "chat": ["start_chat", "proactive invitation shown"],
    "offer": ["sub_offer_click"],
}
ORGANIC = ["organic", "referral", "(none)"]
SOCIAL_SOURCES = ["QxAxdyPLuQMEcrdZWdWb", "MvfHsxITijuriZxsqZqt", "ISrKoXQCxqqYvAZICvjs",
                  "IZEXUFLARCUMynmHNBGo", "PlbkrSYoHuZBWfYjYnfw", "gVRrcxiDQubJiljoTbGm"]
PRESENCE_CITIES = ["Moscow", "Saint Petersburg"]

CAT_RAW = ["utm_source", "utm_medium", "utm_campaign", "utm_adcontent", "utm_keyword",
           "device_category", "device_os", "device_brand", "device_browser",
           "geo_country", "geo_city"]
PAST_COLS = ["utm_source", "utm_medium", "utm_campaign", "utm_adcontent", "utm_keyword",
             "geo_city", "device_brand", "device_browser"]
RECENT_COLS = ["utm_source", "utm_medium", "utm_campaign", "utm_adcontent", "geo_city"]
AGE_COLS = ["utm_campaign", "utm_adcontent", "utm_source"]
WINDOWS = [7, 30]
PAST_SMOOTHING = 200     # same values as in the notebooks
RECENT_SMOOTHING = 100

GROUPS = list(EVENT_GROUPS)
HISTORY_COLS = (["prev_sessions", "prev_conversions", "last_visit_dt", "last_n_hits", "sum_hits",
                 "first_medium"] + [f"past_{g}" for g in GROUPS])


# --- raw input ----------------------------------------------------------------------------

def clean_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    df = sessions.copy()
    for c in CAT_RAW:
        df[c] = df[c].fillna("(missing)")
    df["visit_dt"] = pd.to_datetime(df["visit_dt"])
    df["visit_day"] = df["visit_dt"].dt.normalize()
    return df


def _group_counts(actions: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({g: actions.isin(a).astype("int16") for g, a in EVENT_GROUPS.items()})


def summarise_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """One row per visit: what is known when it finishes (outcome, size, events) and its first 3 events."""
    hits = hits.sort_values(["session_id", "hit_number"])
    counts = _group_counts(hits["event_action"])
    counts["session_id"] = hits["session_id"].values
    counts["is_target"] = hits["event_action"].isin(TARGET_ACTIONS).values

    by_visit = counts.groupby("session_id")
    summary = by_visit[GROUPS].sum()
    summary["target"] = by_visit["is_target"].max().astype(int)
    summary["n_hits"] = by_visit.size()
    summary["landing_path"] = hits.drop_duplicates("session_id").set_index("session_id")["hit_page_path"]

    first3 = counts[hits.groupby("session_id").cumcount().values < 3].groupby("session_id")
    early = first3[GROUPS].sum().add_prefix("early_")
    early["target_in_first3"] = first3["is_target"].max()
    return summary.join(early)


def first_events_summary(first_events: list) -> dict:
    """Early-session features from a visit's first (up to) 3 event actions, as sent by the website."""
    actions = pd.Series(first_events[:3], dtype=object)
    out = {f"early_{g}": int(actions.isin(a).sum()) for g, a in EVENT_GROUPS.items()}
    out["target_in_first3"] = bool(actions.isin(TARGET_ACTIONS).any())
    return out


# --- store 1: category stats ---------------------------------------------------------------

def build_category_stats(history: pd.DataFrame, day) -> tuple:
    """Stats as of the start of `day`, from finished visits before it.

    `history` = cleaned sessions joined with summarise_hits (target, n_hits).
    Conversion rates use only visits with logged events, as in training.
    """
    day = pd.Timestamp(day).normalize()
    before = history[history["visit_day"] < day]
    known = before[before["n_hits"] > 0]

    prior = known["target"].mean()
    priors = {"past": prior}
    rows = []
    for col in PAST_COLS:
        g = known.groupby(col)["target"].agg(["sum", "count"])
        rows.append(pd.DataFrame({
            "column": col, "value": g.index,
            "past_cr": (g["sum"] + PAST_SMOOTHING * prior) / (g["count"] + PAST_SMOOTHING),
            "past_n": g["count"]}))
    stats = pd.concat(rows, ignore_index=True)

    for days in WINDOWS:
        recent = known[known["visit_day"] >= day - pd.Timedelta(days=days)]
        p = recent["target"].mean() if len(recent) else prior
        priors[f"{days}d"] = p
        rows = []
        for col in RECENT_COLS:
            g = recent.groupby(col)["target"].agg(["sum", "count"])
            rows.append(pd.DataFrame({
                "column": col, "value": g.index,
                f"cr_{days}d": (g["sum"] + RECENT_SMOOTHING * p) / (g["count"] + RECENT_SMOOTHING),
                f"n_{days}d": g["count"]}))
        stats = stats.merge(pd.concat(rows, ignore_index=True), on=["column", "value"], how="outer")

    rows = [pd.DataFrame({"column": c, "value": s.index, "first_seen": s.values})
            for c in AGE_COLS for s in [before.groupby(c)["visit_day"].min()]]
    stats = stats.merge(pd.concat(rows, ignore_index=True), on=["column", "value"], how="outer")
    return stats, {"day": str(day.date()), **{k: float(v) for k, v in priors.items()}}


def _lookup(stats: pd.DataFrame, col: str, values: pd.Series, field: str) -> np.ndarray:
    table = stats.loc[stats["column"] == col].set_index("value")[field]
    return values.map(table).to_numpy(dtype="float64" if field != "first_seen" else "datetime64[ns]")


# --- store 2: visitor history ---------------------------------------------------------------

def build_client_history(history: pd.DataFrame) -> pd.DataFrame:
    """One row per client from all their finished visits (with or without logged events)."""
    h = history.sort_values(["client_id", "visit_dt", "visit_number"])
    g = h.groupby("client_id")
    out = pd.DataFrame({
        "prev_sessions": g.size(),
        "prev_conversions": g["target"].sum(),
        "last_visit_dt": g["visit_dt"].last(),
        "last_n_hits": g["n_hits"].last(),
        "sum_hits": g["n_hits"].sum(),
        "first_medium": g["utm_medium"].first(),
    })
    for name in GROUPS:
        out[f"past_{name}"] = g[name].sum()
    return out


def history_before_each_visit(day_visits: pd.DataFrame, clients: pd.DataFrame) -> pd.DataFrame:
    """History row for every visit of one day, including the client's earlier visits that same day.

    `clients` is the history at the start of the day. Used by the batch job, where yesterday's
    visits are all finished; the API gets the same result because the store is updated per visit.
    """
    v = day_visits.reset_index(drop=True)
    order = v.sort_values(["client_id", "visit_dt", "visit_number"]).index
    s = v.loc[order]
    g = s.groupby("client_id", sort=False)
    base = clients.reindex(s["client_id"])
    base.index = s.index

    earlier = pd.DataFrame(index=s.index)
    earlier["n"] = g.cumcount()
    earlier["conv"] = g["target"].cumsum() - s["target"]
    earlier["hits"] = g["n_hits"].cumsum() - s["n_hits"]
    for name in GROUPS:
        earlier[name] = g[name].cumsum() - s[name]
    has_earlier = earlier["n"] > 0

    out = pd.DataFrame(index=s.index)
    out["prev_sessions"] = base["prev_sessions"].fillna(0) + earlier["n"]
    out["prev_conversions"] = base["prev_conversions"].fillna(0) + earlier["conv"]
    out["last_visit_dt"] = g["visit_dt"].shift(1).where(has_earlier, base["last_visit_dt"])
    out["last_n_hits"] = g["n_hits"].shift(1).where(has_earlier, base["last_n_hits"])
    out["sum_hits"] = base["sum_hits"].fillna(0) + earlier["hits"]
    out["first_medium"] = base["first_medium"].fillna(g["utm_medium"].transform("first"))
    for name in GROUPS:
        out[f"past_{name}"] = base[f"past_{name}"].fillna(0) + earlier[name]
    new = out["prev_sessions"] == 0
    out.loc[new, ["last_visit_dt", "last_n_hits", "sum_hits"]] = np.nan
    return out.sort_index()


def visitor_features(visits: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Visitor-history features. `history` has one row per visit, aligned with `visits`."""
    c = history.reset_index(drop=True)
    n = c["prev_sessions"].fillna(0)
    out = pd.DataFrame({
        "prev_sessions": n,
        "prev_conversions": c["prev_conversions"].fillna(0),
        "days_since_prev": (visits["visit_dt"].values - c["last_visit_dt"]).dt.total_seconds() / 86400,
        "prev_session_hits": c["last_n_hits"],
        "mean_prev_hits": c["sum_hits"] / n.replace(0, np.nan),
        "first_medium": c["first_medium"].fillna(pd.Series(visits["utm_medium"].values)),
    })
    out["prev_converted"] = (out["prev_conversions"] > 0).astype(int)
    for name in GROUPS:
        out[f"past_{name}"] = c[f"past_{name}"].fillna(0)
    out["past_any_card_view"] = (out["past_card_view"] > 0).astype(int)
    return out


# --- all features ----------------------------------------------------------------------------

def _landing(paths: pd.Series) -> pd.DataFrame:
    path = (paths.str.split("?").str[0]
                 .str.replace("sberauto.com", "", regex=False)
                 .str.strip("/"))
    parts = path.str.split("/")
    query = paths.str.split("?").str[1].fillna("")
    section = parts.str[0].replace("", "home")
    return pd.DataFrame({
        "landing_section": section.fillna("(missing)"),
        "landing_car_brand": parts.str[1].where(parts.str[0] == "cars", "(none)").fillna("(none)"),
        "landing_depth": parts.str.len(),
        "lp_n_params": query.str.count("="),
        "lp_rental_page": query.str.contains("rental_page").astype(int),
        "lp_city_filter": query.str.contains(r"(?:^|&)city=").astype(int),
        "lp_yandex_click": query.str.contains("yclid").astype(int),
        "lp_facebook_click": query.str.contains("fbclid").astype(int),
        "lp_price_filter": query.str.contains("rent_price").astype(int),
    })


def make_features(visits: pd.DataFrame, stats: pd.DataFrame, priors: dict,
                  history: pd.DataFrame) -> pd.DataFrame:
    """All model features for cleaned visits.

    `visits` needs the raw session columns, `landing_path` and the early_* counts.
    `history` has one row per visit: the client's history *before* that visit.
    """
    v = visits.reset_index(drop=True)
    f = v[["session_id", "visit_number"] + CAT_RAW].copy()

    f["hour"] = v["visit_dt"].dt.hour
    f["weekday"] = v["visit_dt"].dt.weekday
    f["is_weekend"] = (f["weekday"] >= 5).astype(int)
    res = v["device_screen_resolution"].str.extract(r"(\d+)x(\d+)").astype(float)
    f["screen_w"], f["screen_h"] = res[0], res[1]
    f["screen_area"] = f["screen_w"] * f["screen_h"]

    f["is_organic"] = v["utm_medium"].isin(ORGANIC).astype(int)
    f["is_social_ads"] = v["utm_source"].isin(SOCIAL_SOURCES).astype(int)
    f["is_presence_city"] = v["geo_city"].isin(PRESENCE_CITIES).astype(int)
    f["has_campaign"] = (v["utm_campaign"] != "(missing)").astype(int)
    f["has_keyword"] = (v["utm_keyword"] != "(missing)").astype(int)

    for col in AGE_COLS:
        first_seen = pd.Series(_lookup(stats, col, v[col], "first_seen")).fillna(v["visit_day"])
        f[f"{col}_age_days"] = (v["visit_day"] - first_seen).dt.days

    for col in PAST_COLS:
        f[f"{col}_past_cr"] = np.nan_to_num(_lookup(stats, col, v[col], "past_cr"), nan=priors["past"])
        f[f"{col}_past_n"] = np.nan_to_num(_lookup(stats, col, v[col], "past_n"))
    for col in RECENT_COLS:
        for days in WINDOWS:
            f[f"{col}_cr_{days}d"] = np.nan_to_num(_lookup(stats, col, v[col], f"cr_{days}d"),
                                                   nan=priors[f"{days}d"])
            f[f"{col}_n_{days}d"] = np.nan_to_num(_lookup(stats, col, v[col], f"n_{days}d"))
    f["site_cr_7d"] = priors["7d"]

    f = pd.concat([f, visitor_features(v, history), _landing(v["landing_path"])], axis=1)
    early = [c for c in v.columns if c.startswith("early_")]
    f[early] = v[early].to_numpy()
    return f
