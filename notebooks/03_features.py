# %% [markdown]
# # 03. Feature engineering
#
# **Rule for every feature:** it must be known **at the moment the session starts**.
# Anything that happens later in the session (clicks, pages after landing) would leak the answer.
#
# Five feature groups:
# 1. **Raw session attributes**: traffic source, campaign, device, city, time.
# 2. **Generalising traffic features**: traffic type, organic / social flags, how new a campaign is.
# 3. **Historical conversion rates**: how well each source / campaign / city converted *in the past*.
# 4. **Visitor history**: what this visitor did on previous visits.
# 5. **Landing page**: the first page of the visit (which car, which section).

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROCESSED = "../data/processed"

ORGANIC = ["organic", "referral", "(none)"]
SOCIAL_SOURCES = ["QxAxdyPLuQMEcrdZWdWb", "MvfHsxITijuriZxsqZqt", "ISrKoXQCxqqYvAZICvjs",
                  "IZEXUFLARCUMynmHNBGo", "PlbkrSYoHuZBWfYjYnfw", "gVRrcxiDQubJiljoTbGm"]
PRESENCE_CITIES = ["Moscow", "Saint Petersburg"]

# %%
sessions = pd.read_parquet(f"{PROCESSED}/sessions.parquet")
target = pd.read_parquet(f"{PROCESSED}/target.parquet")
hits = pd.read_parquet(f"{PROCESSED}/hits.parquet",
                       columns=["session_id", "hit_number", "hit_page_path"])

df = sessions.merge(target, on="session_id")
df["has_hits"] = df["session_id"].isin(set(hits["session_id"]))
df = df.sort_values("visit_dt").reset_index(drop=True)
df["visit_day"] = df["visit_dt"].dt.normalize()

# %% [markdown]
# ## 1. Raw session attributes

# %%
CAT_RAW = ["utm_source", "utm_medium", "utm_campaign", "utm_adcontent", "utm_keyword",
           "device_category", "device_os", "device_brand", "device_browser",
           "geo_country", "geo_city"]
for c in CAT_RAW:
    df[c] = df[c].fillna("(missing)")

df["hour"] = df["visit_dt"].dt.hour
df["weekday"] = df["visit_dt"].dt.weekday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)

res = df["device_screen_resolution"].str.extract(r"(\d+)x(\d+)").astype(float)
df["screen_w"], df["screen_h"] = res[0], res[1]
df["screen_area"] = df["screen_w"] * df["screen_h"]

# %% [markdown]
# ## 2. Generalising traffic features
# These describe *what kind* of traffic a session is, so they still work for campaigns
# the model has never seen.

# %%
df["is_organic"] = df["utm_medium"].isin(ORGANIC).astype(int)
df["is_social_ads"] = df["utm_source"].isin(SOCIAL_SOURCES).astype(int)
df["is_presence_city"] = df["geo_city"].isin(PRESENCE_CITIES).astype(int)
df["has_campaign"] = (df["utm_campaign"] != "(missing)").astype(int)
df["has_keyword"] = (df["utm_keyword"] != "(missing)").astype(int)

# How long a campaign / creative / source has existed when the session starts.
# A brand-new campaign gets 0, which the model can learn from.
for c in ["utm_campaign", "utm_adcontent", "utm_source"]:
    first_seen = df.groupby(c)["visit_day"].transform("min")
    df[f"{c}_age_days"] = (df["visit_day"] - first_seen).dt.days

# %% [markdown]
# ## 3. Historical conversion rates (time-aware target encoding)
# For each category we compute its conversion rate using **only days strictly before** the session.
# Sessions without logged events are excluded from these statistics (their label is unknown).
# Rates are smoothed towards the overall rate, so a category with 3 past sessions does not get 0% or 100%.

# %%
SMOOTHING = 200
known = df[df["has_hits"]]

def past_rate(col):
    daily = known.groupby([col, "visit_day"])["target"].agg(["sum", "count"]).reset_index()
    daily = daily.sort_values([col, "visit_day"])
    daily["past_conv"] = daily.groupby(col)["sum"].cumsum() - daily["sum"]
    daily["past_n"] = daily.groupby(col)["count"].cumsum() - daily["count"]

    total = known.groupby("visit_day")["target"].agg(["sum", "count"]).sort_index()
    prior = (total["sum"].cumsum() - total["sum"]) / (total["count"].cumsum() - total["count"])
    daily["prior"] = daily["visit_day"].map(prior).fillna(known["target"].mean())

    daily[f"{col}_past_cr"] = ((daily["past_conv"] + SMOOTHING * daily["prior"])
                               / (daily["past_n"] + SMOOTHING))
    daily[f"{col}_past_n"] = daily["past_n"]

    out = df[[col, "visit_day"]].merge(
        daily[[col, "visit_day", f"{col}_past_cr", f"{col}_past_n"]],
        on=[col, "visit_day"], how="left")
    # a category seen for the first time today has no past: fall back to the prior
    out[f"{col}_past_cr"] = out[f"{col}_past_cr"].fillna(df["visit_day"].map(prior)).values
    return out[[f"{col}_past_cr", f"{col}_past_n"]].fillna(0).values

for c in ["utm_source", "utm_medium", "utm_campaign", "utm_adcontent", "utm_keyword",
          "geo_city", "device_brand", "device_browser"]:
    df[[f"{c}_past_cr", f"{c}_past_n"]] = past_rate(c)

# %% [markdown]
# ## 4. Visitor history
# Everything here comes from the visitor's **previous** sessions, which are finished
# before the current one starts.

# %%
hits_per_session = hits.groupby("session_id").size()
df["n_hits"] = df["session_id"].map(hits_per_session).fillna(0)

df = df.sort_values(["client_id", "visit_dt"])
g = df.groupby("client_id")

df["prev_sessions"] = g.cumcount()
df["prev_conversions"] = g["target"].cumsum() - df["target"]
df["prev_converted"] = (df["prev_conversions"] > 0).astype(int)
df["days_since_prev"] = g["visit_dt"].diff().dt.total_seconds() / 86400
df["prev_session_hits"] = g["n_hits"].shift(1)
df["mean_prev_hits"] = (g["n_hits"].cumsum() - df["n_hits"]) / df["prev_sessions"].replace(0, np.nan)
df["first_medium"] = g["utm_medium"].transform("first")
df = df.sort_values("visit_dt").reset_index(drop=True)

# %% [markdown]
# ## 5. Landing page
# The first page of the visit is known as soon as the visitor arrives.
# It tells us whether they landed on the home page, a car catalogue, a specific car, or a promo page.

# %%
first_hit = (hits.sort_values(["session_id", "hit_number"])
                 .drop_duplicates("session_id")
                 .set_index("session_id")["hit_page_path"])
path = (first_hit.str.split("?").str[0]
                 .str.replace("sberauto.com", "", regex=False)
                 .str.strip("/"))
parts = path.str.split("/")

landing = pd.DataFrame({
    "landing_section": parts.str[0].replace("", "home"),
    "landing_car_brand": np.where(parts.str[0] == "cars", parts.str[1], "(none)"),
    "landing_depth": parts.str.len(),
})
landing["landing_car_brand"] = landing["landing_car_brand"].fillna("(none)")
df = df.join(landing, on="session_id")
for c in ["landing_section", "landing_car_brand"]:
    df[c] = df[c].fillna("(missing)")

# %% [markdown]
# ## Save
# Only sessions with logged events are modelled.

# %%
model_df = df[df["has_hits"]].drop(columns=["has_hits", "n_hits"])
model_df.to_parquet(f"{PROCESSED}/features.parquet", index=False)
print(model_df.shape)
model_df.filter(regex="past_cr|prev_|age_days|landing").describe().T.round(3)
