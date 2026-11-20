# %% [markdown]
# # 02. Validation design
#
# **Goal:** decide how to split the data so that the test score is honest , 
# i.e. close to what the model would achieve on new, future traffic.
#
# We train the *same* simple model under three splits and compare the scores.

# %%
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

PROCESSED = "../data/processed"

CAT_COLS = ["utm_source", "utm_medium", "utm_campaign", "utm_adcontent", "utm_keyword",
            "device_category", "device_os", "device_brand", "device_screen_resolution",
            "device_browser", "geo_country", "geo_city"]
NUM_COLS = ["visit_number", "hour", "weekday"]

# %% [markdown]
# ## 1. Modelling table
# Sessions with no logged events are excluded (Step 1): their label is unknown.

# %%
sessions = pd.read_parquet(f"{PROCESSED}/sessions.parquet")
target = pd.read_parquet(f"{PROCESSED}/target.parquet")
hit_ids = set(pd.read_parquet(f"{PROCESSED}/hits.parquet", columns=["session_id"])["session_id"])

df = sessions.merge(target, on="session_id")
df = df[df["session_id"].isin(hit_ids)].copy()

df["hour"] = df["visit_dt"].dt.hour
df["weekday"] = df["visit_dt"].dt.weekday
for c in CAT_COLS:
    df[c] = df[c].fillna("(missing)").astype("category")

print(f"modelling rows: {len(df):,} | conversion rate: {df['target'].mean():.2%}")

# %% [markdown]
# ## 2. One model, three ways to split

# %%
def fit_score(train, test):
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                               min_child_samples=100, verbose=-1, random_state=42)
    model.fit(train[CAT_COLS + NUM_COLS], train["target"])
    proba = model.predict_proba(test[CAT_COLS + NUM_COLS])[:, 1]
    return roc_auc_score(test["target"], proba)

results = {}

# A. Random split — the most common default
tr, te = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)
results["A. random split"] = (fit_score(tr, te), len(te), te["target"].mean())
overlap = te["client_id"].isin(set(tr["client_id"])).mean()
print(f"A: share of test sessions whose visitor is also in train: {overlap:.1%}")

# B. Random split by visitor — a visitor is entirely in train or entirely in test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
i_tr, i_te = next(gss.split(df, groups=df["client_id"]))
tr, te = df.iloc[i_tr], df.iloc[i_te]
results["B. random split by visitor"] = (fit_score(tr, te), len(te), te["target"].mean())

# C. Time split — train on the past, test on the future
train = df[df["visit_dt"] < "2021-11-01"]
valid = df[(df["visit_dt"] >= "2021-11-01") & (df["visit_dt"] < "2021-12-01")]
test = df[df["visit_dt"] >= "2021-12-01"]
results["C. time split (test = December)"] = (fit_score(train, test), len(test), test["target"].mean())

print(f"\ntime split sizes: train {len(train):,} (May–Oct) | valid {len(valid):,} (Nov) | test {len(test):,} (Dec)")

# %%
summary = pd.DataFrame(results, index=["roc_auc", "test_rows", "test_cr"]).T
summary["roc_auc"] = summary["roc_auc"].astype(float).round(4)
summary["test_rows"] = summary["test_rows"].astype(int)
summary["test_cr"] = (summary["test_cr"].astype(float) * 100).round(2)
summary

# %% [markdown]
# ## 3. Decision
# * **Train:** May–October · **Validation:** November (tuning, early stopping, threshold) · **Test:** December (touched once, at the end).
# * Returning visitors in December are allowed: in production we *would* know their past visits.
#   What we must guarantee instead is that every feature uses **only information available before the session starts**.
