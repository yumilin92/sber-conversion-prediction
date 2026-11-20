# %% [markdown]
# # 01. Data audit
#
# **Goal:** understand what one row means in each table, check data quality, and define the target.
# Nothing is modelled here. Everything downstream depends on getting this step right.

# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

RAW = "../data/raw"
PROCESSED = "../data/processed"

TARGET_ACTIONS = [
    "sub_car_claim_click", "sub_car_claim_submit_click", "sub_open_dialog_click",
    "sub_custom_question_submit_click", "sub_call_number_click",
    "sub_callback_submit_click", "sub_submit_success", "sub_car_request_submit_click",
]

# %% [markdown]
# ## 1. Load once, save as Parquet
# The raw CSVs are 0.4 GB (sessions) and 4.3 GB (hits). Parquet is columnar and compressed,
# so later notebooks read only the columns they need in seconds.

# %%
sessions = pd.read_csv(f"{RAW}/ga_sessions.csv", dtype=str)
sessions["visit_number"] = sessions["visit_number"].astype(int)
sessions["visit_dt"] = pd.to_datetime(sessions["visit_date"] + " " + sessions["visit_time"])

hits = pd.read_csv(
    f"{RAW}/ga_hits.csv",
    usecols=["session_id", "hit_date", "hit_number", "hit_type", "hit_page_path",
             "event_category", "event_action"],
    dtype={"hit_number": "int32"},
)

sessions.to_parquet(f"{PROCESSED}/sessions.parquet", index=False)
hits.to_parquet(f"{PROCESSED}/hits.parquet", index=False)
print("sessions:", sessions.shape, "| hits:", hits.shape)

# %% [markdown]
# ## 2. Grain, what is one row?
# * `sessions`: one row = one visit (`session_id`). A visitor (`client_id`) can have many visits.
# * `hits`: one row = one event inside a visit.

# %%
print("unique session_id :", sessions["session_id"].nunique(), "of", len(sessions))
print("unique client_id  :", sessions["client_id"].nunique())
print("period            :", sessions["visit_dt"].min(), "→", sessions["visit_dt"].max())
print("hits per session  :", round(len(hits) / hits["session_id"].nunique(), 1))
print("\nsessions per client:")
print(sessions.groupby("client_id").size().describe(percentiles=[.5, .9, .99]).round(1))

# %% [markdown]
# ## 3. Do the tables match?

# %%
s_ids = set(sessions["session_id"])
h_ids = set(hits["session_id"])
print("sessions with no hits :", len(s_ids - h_ids))
print("hit sessions not in sessions table:", len(h_ids - s_ids))

# %% [markdown]
# ## 4. Missing and placeholder values
# Google Analytics writes `(not set)` instead of an empty value, so NaN alone underestimates gaps.

# %%
quality = pd.DataFrame({
    "nan_%": sessions.isna().mean() * 100,
    "not_set_%": (sessions == "(not set)").mean() * 100,
    "n_unique": sessions.nunique(),
}).round(2).sort_values("nan_%", ascending=False)
quality

# %% [markdown]
# ## 5. Target
# A session is a **conversion** if it has at least one of the 8 target events
# ("leave a request", "order a call" …). Several target events in one visit still count as one.

# %%
is_target = hits["event_action"].isin(TARGET_ACTIONS)
print("target events by type:")
print(hits.loc[is_target, "event_action"].value_counts())

converted = set(hits.loc[is_target, "session_id"])
sessions["target"] = sessions["session_id"].isin(converted).astype(int)
print(f"\nconversion rate: {sessions['target'].mean():.2%}  "
      f"({sessions['target'].sum():,} of {len(sessions):,} sessions)")

# %%
sessions[["session_id", "target"]].to_parquet(f"{PROCESSED}/target.parquet", index=False)

# %% [markdown]
# ## 6. Conversion over time
# Checks for logging breaks or seasonality before we choose how to split train and test.

# %%
weekly = (sessions.set_index("visit_dt")
          .resample("W")["target"].agg(["size", "mean"])
          .rename(columns={"size": "sessions", "mean": "cr"}))
weekly["cr"] = (weekly["cr"] * 100).round(2)
weekly
