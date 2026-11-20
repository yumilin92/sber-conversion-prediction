# %% [markdown]
# # 04. Experiments: which features actually help?
#
# Train on **May–October**, evaluate on **November** (validation).
# December stays untouched until the final model is chosen.
#
# We add feature groups one at a time, so we can see the contribution of each.

# %%
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

PROCESSED = "../data/processed"
df = pd.read_parquet(f"{PROCESSED}/features.parquet")

train = df[df["visit_dt"] < "2021-11-01"].copy()
valid = df[(df["visit_dt"] >= "2021-11-01") & (df["visit_dt"] < "2021-12-01")].copy()
print(f"train {len(train):,} | valid {len(valid):,} | valid CR {valid['target'].mean():.2%}")

# %%
RAW_IDS = ["utm_source", "utm_campaign", "utm_adcontent", "utm_keyword"]
RAW_OTHER = ["utm_medium", "device_category", "device_os", "device_brand", "device_browser",
             "geo_country", "geo_city", "visit_number", "hour", "weekday"]
GENERAL = ["is_weekend", "screen_w", "screen_h", "screen_area", "is_organic", "is_social_ads",
           "is_presence_city", "has_campaign", "has_keyword",
           "utm_campaign_age_days", "utm_adcontent_age_days", "utm_source_age_days"]
PAST_CR = [c for c in df.columns if c.endswith("_past_cr") or c.endswith("_past_n")]
VISITOR = ["prev_sessions", "prev_conversions", "prev_converted", "days_since_prev",
           "prev_session_hits", "mean_prev_hits", "first_medium"]
LANDING = ["landing_section", "landing_car_brand", "landing_depth"]

ALL_CATS = RAW_IDS + ["utm_medium", "device_category", "device_os", "device_brand",
                      "device_browser", "geo_country", "geo_city", "first_medium",
                      "landing_section", "landing_car_brand"]
for c in ALL_CATS:
    cats = pd.CategoricalDtype(train[c].unique())          # categories known in training only
    train[c] = train[c].astype(cats)
    valid[c] = valid[c].astype(cats)                       # unseen values become NaN

PARAMS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
              cat_smooth=50, min_data_per_group=200, verbose=-1, random_state=42)

def run(features):
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(train[features], train["target"],
              eval_set=[(valid[features], valid["target"])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    auc = roc_auc_score(valid["target"], model.predict_proba(valid[features])[:, 1])
    return round(auc, 4), model.best_iteration_

# %%
experiments = {
    "E0 raw session attributes":            RAW_IDS + RAW_OTHER,
    "E1 + generalising traffic features":   RAW_IDS + RAW_OTHER + GENERAL,
    "E2 + historical conversion rates":     RAW_IDS + RAW_OTHER + GENERAL + PAST_CR,
    "E3 + visitor history":                 RAW_IDS + RAW_OTHER + GENERAL + PAST_CR + VISITOR,
    "E4 + landing page":                    RAW_IDS + RAW_OTHER + GENERAL + PAST_CR + VISITOR + LANDING,
    "E5 E4 without raw campaign/ad IDs":    RAW_OTHER + GENERAL + PAST_CR + VISITOR + LANDING,
}
rows = []
for name, feats in experiments.items():
    auc, it = run(feats)
    rows.append((name, len(feats), auc, it))
    print(f"{name:40s} features={len(feats):3d}  valid AUC={auc}  trees={it}")

results = pd.DataFrame(rows, columns=["experiment", "n_features", "valid_auc", "best_trees"])
results.to_csv("../reports/experiments.csv", index=False)
results

# %% [markdown]
# ## Follow-up experiments on the best set (E5)

# %%
E5 = RAW_OTHER + GENERAL + PAST_CR + VISITOR + LANDING
follow_up = {
    "E6 E5 without raw city/brand/browser": [c for c in E5 if c not in ["geo_city", "device_brand", "device_browser"]],
    "E7 E5 without past_n counts": [c for c in E5 if not c.endswith("_past_n")],
}
for name, feats in follow_up.items():
    print(f"{name:40s} valid AUC, trees = {run(feats)}")

# %% [markdown]
# ## Light tuning of tree size on E5

# %%
for leaves, min_child in [(15, 1000), (31, 500), (63, 200), (127, 200)]:
    PARAMS.update(num_leaves=leaves, min_child_samples=min_child)
    print(f"num_leaves={leaves:3d} min_child_samples={min_child:4d}  valid AUC, trees = {run(E5)}")
