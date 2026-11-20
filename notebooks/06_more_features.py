# %% [markdown]
# # 06. More features
#
# Three new ideas, all still known **when the session starts**:
# 1. **Recent conversion rates** (last 7 / 30 days). Traffic quality drifts, so a campaign's
#    conversion rate since May can be stale. Recent windows react to drift, and to new campaigns
#    within days of launch.
# 2. **What the visitor did on previous visits**: viewed car cards, used search, opened chat…
#    Previous visits are finished before the current one starts.
# 3. **Landing URL parameters**: ad click IDs, rental-page flags, city filter.
#
# Plus features for a second use case, an **early-session model** that scores a visit after its
# first 3 events (e.g. to decide whether to open a chat). Kept separate from the session-start model.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROCESSED = "../data/processed"
TARGET_ACTIONS = [
    "sub_car_claim_click", "sub_car_claim_submit_click", "sub_open_dialog_click",
    "sub_custom_question_submit_click", "sub_call_number_click",
    "sub_callback_submit_click", "sub_submit_success", "sub_car_request_submit_click",
]
EVENT_GROUPS = {
    "card_view":   ["view_card", "view_new_card", "go_to_car_card", "sub_car_page"],
    "search":      ["search_form_region", "search_form_search_btn", "search_form_mark_select",
                    "search_form_model_select", "search_form_search_car_type_select",
                    "sap_search_form_cost_to", "sap_search_form_cost_from"],
    "photos":      ["photos", "photos_all"],
    "catalogue":   ["sub_view_cars_click", "pagination_click", "view_more_click", "listing_ads_sort"],
    "quiz":        ["quiz_show"],
    "chat":        ["start_chat", "proactive invitation shown"],
    "offer":       ["sub_offer_click"],
}

df = pd.read_parquet(f"{PROCESSED}/features.parquet")
hits = pd.read_parquet(f"{PROCESSED}/hits.parquet",
                       columns=["session_id", "hit_number", "hit_page_path", "event_action"])
sessions = pd.read_parquet(f"{PROCESSED}/sessions.parquet", columns=["session_id", "client_id", "visit_dt"])
target = pd.read_parquet(f"{PROCESSED}/target.parquet")

# %% [markdown]
# ## 1. Recent conversion rates (rolling windows)
# Daily conversions and sessions per category to rolling sum over the previous N days (excluding today)
# to smoothed rate. A new campaign gets a real rate after its first full day.

# %%
SMOOTHING = 100

def rolling_rate(col, days):
    daily = df.groupby([col, "visit_day"])["target"].agg(["sum", "count"])
    days_index = pd.date_range(df["visit_day"].min(), df["visit_day"].max(), freq="D")
    wide_sum = daily["sum"].unstack(col).reindex(days_index).fillna(0)
    wide_cnt = daily["count"].unstack(col).reindex(days_index).fillna(0)
    past_sum = wide_sum.rolling(days, min_periods=1).sum().shift(1).fillna(0)
    past_cnt = wide_cnt.rolling(days, min_periods=1).sum().shift(1).fillna(0)

    total_sum = wide_sum.sum(axis=1).rolling(days, min_periods=1).sum().shift(1)
    total_cnt = wide_cnt.sum(axis=1).rolling(days, min_periods=1).sum().shift(1)
    prior = (total_sum / total_cnt).fillna(df["target"].mean())

    rate = past_sum.add(SMOOTHING * prior, axis=0).div(past_cnt + SMOOTHING)
    long = rate.stack().rename(f"{col}_cr_{days}d").reset_index()
    long.columns = ["visit_day", col, f"{col}_cr_{days}d"]
    cnt = past_cnt.stack().rename(f"{col}_n_{days}d").reset_index()
    cnt.columns = ["visit_day", col, f"{col}_n_{days}d"]
    return long.merge(cnt, on=["visit_day", col]), prior

for col in ["utm_source", "utm_medium", "utm_campaign", "utm_adcontent", "geo_city"]:
    for days in [7, 30]:
        table, prior = rolling_rate(col, days)
        df = df.merge(table, on=["visit_day", col], how="left")

# overall site conversion over the last 7 days — captures the general trend
_, prior7 = rolling_rate("utm_medium", 7)
df["site_cr_7d"] = df["visit_day"].map(prior7)

# %% [markdown]
# ## 2. What the visitor did on previous visits

# %%
for name, actions in EVENT_GROUPS.items():
    hits[name] = hits["event_action"].isin(actions).astype("int16")
per_session = hits.groupby("session_id")[list(EVENT_GROUPS)].sum()

hist = sessions.merge(target, on="session_id").join(per_session, on="session_id")
hist[list(EVENT_GROUPS)] = hist[list(EVENT_GROUPS)].fillna(0)
hist = hist.sort_values(["client_id", "visit_dt"])
g = hist.groupby("client_id")
for name in EVENT_GROUPS:
    hist[f"past_{name}"] = g[name].cumsum() - hist[name]
hist["past_any_card_view"] = (hist["past_card_view"] > 0).astype(int)
df = df.merge(hist[["session_id"] + [f"past_{n}" for n in EVENT_GROUPS] + ["past_any_card_view"]],
              on="session_id", how="left")

# %% [markdown]
# ## 3. Landing URL parameters

# %%
first_hit = (hits.sort_values(["session_id", "hit_number"])
                 .drop_duplicates("session_id")
                 .set_index("session_id")["hit_page_path"])
query = first_hit.str.split("?").str[1].fillna("")
landing_params = pd.DataFrame({
    "lp_n_params": query.str.count("=").astype("int16"),
    "lp_rental_page": query.str.contains("rental_page").astype("int8"),
    "lp_city_filter": query.str.contains(r"(?:^|&)city=").astype("int8"),
    "lp_yandex_click": query.str.contains("yclid").astype("int8"),
    "lp_facebook_click": query.str.contains("fbclid").astype("int8"),
    "lp_price_filter": query.str.contains("rent_price").astype("int8"),
})
df = df.join(landing_params, on="session_id")

# %% [markdown]
# ## 4. Early-session features (second use case)
# The first 3 events of the visit. Sessions where a target action happens within those 3 events
# are flagged and excluded from the early-session model, there would be nothing left to predict.

# %%
first3 = hits.sort_values(["session_id", "hit_number"]).groupby("session_id").head(3).copy()
first3["is_target"] = first3["event_action"].isin(TARGET_ACTIONS)
early = first3.groupby("session_id").agg(
    target_in_first3=("is_target", "max"),
    **{f"early_{n}": (n, "sum") for n in EVENT_GROUPS},
)
df = df.join(early, on="session_id")

# %%
df.to_parquet(f"{PROCESSED}/features_v2.parquet", index=False)
print(df.shape)
df.filter(regex="_7d|_30d|past_card|lp_|early_").describe().T[["mean", "max"]].round(3).head(30)
