# %% [markdown]
# # 07. New features, other models, ensembles
#
# Same protocol as before: train May–October, select on **November**, December untouched.

# %%
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

exec(open("04_experiments.py").read().split("# %%\nexperiments")[0])   # feature lists, PARAMS, run()

df = pd.read_parquet(f"{PROCESSED}/features_v2.parquet")
E5 = RAW_OTHER + GENERAL + PAST_CR + VISITOR + LANDING
ROLLING = [c for c in df.columns if c.endswith(("_7d", "_30d"))]
PAST_BEHAVIOUR = [c for c in df.columns if c.startswith("past_") and not c.endswith(("_past_cr", "_past_n"))]
LP_PARAMS = [c for c in df.columns if c.startswith("lp_")]
EARLY = [c for c in df.columns if c.startswith("early_")]

CATS = ["utm_medium", "device_category", "device_os", "device_brand", "device_browser",
        "geo_country", "geo_city", "first_medium", "landing_section", "landing_car_brand"]

def split(frame):
    tr = frame[frame["visit_dt"] < "2021-11-01"].copy()
    va = frame[(frame["visit_dt"] >= "2021-11-01") & (frame["visit_dt"] < "2021-12-01")].copy()
    for c in CATS:
        cats = pd.CategoricalDtype(tr[c].unique())
        tr[c], va[c] = tr[c].astype(cats), va[c].astype(cats)
    return tr, va

train, valid = split(df)
PARAMS.update(num_leaves=127, min_child_samples=200)

def lgb_fit(features, weights=None, tr=None, va=None):
    tr = train if tr is None else tr
    va = valid if va is None else va
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(tr[features], tr["target"], sample_weight=weights,
              eval_set=[(va[features], va["target"])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    p = model.predict_proba(va[features])[:, 1]
    return roc_auc_score(va["target"], p), model.best_iteration_, p

# %% [markdown]
# ## 1. New feature groups (LightGBM)

# %%
sets = {
    "F0 E5 (previous best)":            E5,
    "F1 + recent conversion rates":     E5 + ROLLING,
    "F2 + previous-visit behaviour":    E5 + ROLLING + PAST_BEHAVIOUR,
    "F3 + landing URL parameters":      E5 + ROLLING + PAST_BEHAVIOUR + LP_PARAMS,
}
lgb_preds = {}
for name, feats in sets.items():
    auc, it, p = lgb_fit(feats)
    lgb_preds[name] = p
    print(f"{name:35s} features={len(feats):3d}  valid AUC={auc:.4f}  trees={it}")

# %% [markdown]
# ## 2. Weight recent months more
# Traffic drifts, so recent sessions describe November better than May does.
# Weight = 0.5 ** (age in days / half-life).

# %%
BEST = sets["F3 + landing URL parameters"]
age_days = (train["visit_dt"].max() - train["visit_dt"]).dt.days
for half_life in [30, 60, 120]:
    w = 0.5 ** (age_days / half_life)
    auc, it, _ = lgb_fit(BEST, weights=w)
    print(f"half-life {half_life:3d} days  valid AUC={auc:.4f}  trees={it}")

# %% [markdown]
# ## 3. Other models on the same features

# %%
results = {"LightGBM": roc_auc_score(valid["target"], lgb_preds["F3 + landing URL parameters"])}

xgb_model = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=8, min_child_weight=50,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=5, tree_method="hist",
                              enable_categorical=True, max_cat_to_onehot=1, eval_metric="auc",
                              early_stopping_rounds=100, n_jobs=-1, random_state=42)
xgb_model.fit(train[BEST], train["target"], eval_set=[(valid[BEST], valid["target"])], verbose=False)
p_xgb = xgb_model.predict_proba(valid[BEST])[:, 1]
results["XGBoost"] = roc_auc_score(valid["target"], p_xgb)

cat_cols = [c for c in BEST if c in CATS]
to_str = lambda f: f[BEST].assign(**{c: f[c].astype(object).fillna("(unseen)").astype(str) for c in cat_cols})
cb = CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=8, l2_leaf_reg=5,
                        eval_metric="AUC", od_type="Iter", od_wait=150, thread_count=-1,
                        random_seed=42, verbose=False)
cb.fit(to_str(train), train["target"], cat_features=cat_cols, eval_set=(to_str(valid), valid["target"]))
p_cb = cb.predict_proba(to_str(valid))[:, 1]
results["CatBoost"] = roc_auc_score(valid["target"], p_cb)

blend = (rankdata(lgb_preds["F3 + landing URL parameters"]) + rankdata(p_xgb) + rankdata(p_cb)) / 3
results["Blend (rank average of 3)"] = roc_auc_score(valid["target"], blend)

for k, v in results.items():
    print(f"{k:28s} valid AUC={v:.4f}")
print("best iterations — XGBoost:", xgb_model.best_iteration, "| CatBoost:", cb.get_best_iteration())

# %% [markdown]
# ## 4. Second use case: early-session model
# Score the visit after its first 3 events. Sessions with a target action inside those 3 events
# are removed (nothing left to predict). Features: everything above + what happened in the first 3 events.

# %%
early_df = df[df["target_in_first3"] == 0]
tr_e, va_e = split(early_df)
auc_e, it_e, _ = lgb_fit(BEST + EARLY, tr=tr_e, va=va_e)
auc_e0, _, _ = lgb_fit(BEST, tr=tr_e, va=va_e)
print(f"early-session rows: train {len(tr_e):,} valid {len(va_e):,} | valid CR {va_e['target'].mean():.2%}")
print(f"same sessions, session-start features only: valid AUC={auc_e0:.4f}")
print(f"session-start + first 3 events:           valid AUC={auc_e:.4f}  trees={it_e}")
