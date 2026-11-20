# %% [markdown]
# # 10. Hypothesis tests for the product team
#
# **What:** four questions the business asked, answered with statistics.
# **Why:** a model ranks visitors; these tests answer "is group A really better than group B, and by how much?"
#
# For each question: conversion rate per group, the difference with a 95% confidence interval,
# a two-proportion z-test, and the same check at **client level** (one row per client),
# because sessions of the same client are not independent.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

df = pd.read_parquet("../data/processed/features_v2.parquet",
                     columns=["client_id", "target", "utm_medium", "device_category",
                              "is_organic", "is_social_ads", "is_presence_city"])
print(f"{len(df):,} sessions | CR {df['target'].mean():.2%}")

def compare(frame, mask_a, mask_b, label_a, label_b):
    a, b = frame.loc[mask_a, "target"], frame.loc[mask_b, "target"]
    pa, pb = a.mean(), b.mean()
    diff = pa - pb
    se = np.sqrt(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b))
    _, p = proportions_ztest([a.sum(), b.sum()], [len(a), len(b)])
    return {"A": label_a, "B": label_b, "n_A": len(a), "n_B": len(b),
            "CR_A_%": round(pa * 100, 2), "CR_B_%": round(pb * 100, 2),
            "diff_pp": round(diff * 100, 2),
            "CI95_pp": f"[{(diff - 1.96 * se) * 100:.2f}, {(diff + 1.96 * se) * 100:.2f}]",
            "relative_%": round(diff / pb * 100, 1), "p_value": f"{p:.1e}"}

def client_level(frame):
    """One row per client: did the client ever convert? Group = the client's most frequent value."""
    g = frame.groupby("client_id")
    out = g["target"].max().to_frame()
    for c in ["is_organic", "is_social_ads", "is_presence_city"]:
        out[c] = g[c].mean().round()
    out["device_category"] = g["device_category"].agg(lambda s: s.mode().iat[0])
    out["paid"] = 1 - out["is_organic"]
    return out

paid = df["is_organic"] == 0

QUESTIONS = [
    ("Organic vs paid traffic",       lambda f: (f["is_organic"] == 1, f["is_organic"] == 0, "organic", "paid")),
    ("Mobile vs desktop",             lambda f: (f["device_category"] == "mobile", f["device_category"] == "desktop", "mobile", "desktop")),
    ("Moscow & SPb vs other cities",  lambda f: (f["is_presence_city"] == 1, f["is_presence_city"] == 0, "Moscow/SPb", "other")),
    ("Social ads vs other paid",      lambda f: ((f["is_social_ads"] == 1) & (f["is_organic"] == 0),
                                                 (f["is_social_ads"] == 0) & (f["is_organic"] == 0), "social ads", "other paid")),
]

# %% [markdown]
# ## 1. Session level

# %%
session_res = pd.DataFrame([{"question": q, **compare(df, *fn(df))} for q, fn in QUESTIONS])
session_res

# %% [markdown]
# ## 2. Client level (robustness check)

# %%
cl = client_level(df)
client_res = pd.DataFrame([{"question": q, **compare(cl, *fn(cl))} for q, fn in QUESTIONS])
client_res[["question", "CR_A_%", "CR_B_%", "diff_pp", "CI95_pp", "relative_%", "p_value"]]

# %% [markdown]
# ## 3. Is mobile vs desktop just a traffic-mix effect?
# Mobile users may come from different channels. Compare devices **within** each medium.

# %%
top_media = df["utm_medium"].value_counts().index[:5]
rows = []
for m in top_media:
    f = df[df["utm_medium"] == m]
    r = compare(f, f["device_category"] == "mobile", f["device_category"] == "desktop", "mobile", "desktop")
    rows.append({"medium": m, "n": len(f), "CR_mobile_%": r["CR_A_%"], "CR_desktop_%": r["CR_B_%"],
                 "diff_pp": r["diff_pp"], "CI95_pp": r["CI95_pp"]})
pd.DataFrame(rows)

# %%
session_res.to_csv("../reports/hypotheses_session.csv", index=False)
client_res.to_csv("../reports/hypotheses_client.csv", index=False)
