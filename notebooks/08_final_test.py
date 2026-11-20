# %% [markdown]
# # 08. Final December test for both use cases
#
# Selected on November:
# * **Session-start model**: feature set F3, rank-average blend of LightGBM + XGBoost + CatBoost.
# * **Early-session model**: F3 + first-3-events features, LightGBM.
#
# Retrain on May–November with the November best iterations (+20% for the larger training set),
# then score December **once**.

# %%
import warnings
warnings.filterwarnings("ignore")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

exec(open("04_experiments.py").read().split("# %%\nexperiments")[0])

df = pd.read_parquet(f"{PROCESSED}/features_v2.parquet")
E5 = RAW_OTHER + GENERAL + PAST_CR + VISITOR + LANDING
ROLLING = [c for c in df.columns if c.endswith(("_7d", "_30d"))]
PAST_BEHAVIOUR = [c for c in df.columns if c.startswith("past_") and not c.endswith(("_past_cr", "_past_n"))]
LP_PARAMS = [c for c in df.columns if c.startswith("lp_")]
EARLY = [c for c in df.columns if c.startswith("early_")]
F3 = E5 + ROLLING + PAST_BEHAVIOUR + LP_PARAMS
CATS = ["utm_medium", "device_category", "device_os", "device_brand", "device_browser",
        "geo_country", "geo_city", "first_medium", "landing_section", "landing_car_brand"]

def split(frame):
    fit = frame[frame["visit_dt"] < "2021-12-01"].copy()
    test = frame[frame["visit_dt"] >= "2021-12-01"].copy()
    for c in CATS:
        cats = pd.CategoricalDtype(fit[c].unique())
        fit[c], test[c] = fit[c].astype(cats), test[c].astype(cats)
    return fit, test

def decile_table(y, score):
    t = pd.DataFrame({"y": y.values, "score": score})
    t["decile"] = pd.qcut(rankdata(-t["score"], method="ordinal"), 10, labels=range(1, 11))
    d = t.groupby("decile", observed=True)["y"].agg(cr="mean", conversions="sum")
    d["lift"] = (d["cr"] / t["y"].mean()).round(2)
    d["cum_share_of_conversions_%"] = (d["conversions"].cumsum() / d["conversions"].sum() * 100).round(1)
    d["cr_%"] = (d["cr"] * 100).round(2)
    return d[["cr_%", "lift", "cum_share_of_conversions_%"]]

LGB_PARAMS = {**PARAMS, "num_leaves": 127, "min_child_samples": 200}

# %% [markdown]
# ## 1. Session-start model (blend)

# %%
fit, test = split(df)
cat_cols = [c for c in F3 if c in CATS]
to_str = lambda f: f[F3].assign(**{c: f[c].astype(object).fillna("(unseen)").astype(str) for c in cat_cols})

m_lgb = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": 180}).fit(fit[F3], fit["target"])
m_xgb = xgb.XGBClassifier(n_estimators=640, learning_rate=0.03, max_depth=8, min_child_weight=50,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=5, tree_method="hist",
                          enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1,
                          random_state=42).fit(fit[F3], fit["target"])
m_cb = CatBoostClassifier(iterations=520, learning_rate=0.05, depth=8, l2_leaf_reg=5, thread_count=-1,
                          random_seed=42, verbose=False).fit(to_str(fit), fit["target"], cat_features=cat_cols)

p = {"LightGBM": m_lgb.predict_proba(test[F3])[:, 1],
     "XGBoost": m_xgb.predict_proba(test[F3])[:, 1],
     "CatBoost": m_cb.predict_proba(to_str(test))[:, 1]}
p["Blend"] = (rankdata(p["LightGBM"]) + rankdata(p["XGBoost"]) + rankdata(p["CatBoost"])) / (3 * len(test))

for k, v in p.items():
    print(f"{k:10s} December ROC-AUC={roc_auc_score(test['target'], v):.4f}  "
          f"PR-AUC={average_precision_score(test['target'], v):.4f}")
print(f"random PR-AUC = {test['target'].mean():.4f}")
decile_table(test["target"], p["Blend"])

# %% [markdown]
# ## 2. Early-session model

# %%
fit_e, test_e = split(df[df["target_in_first3"] == 0])
m_early = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": 300}).fit(fit_e[F3 + EARLY], fit_e["target"])
p_e = m_early.predict_proba(test_e[F3 + EARLY])[:, 1]
print(f"Early-session December ROC-AUC={roc_auc_score(test_e['target'], p_e):.4f}  "
      f"PR-AUC={average_precision_score(test_e['target'], p_e):.4f}  (random {test_e['target'].mean():.4f})")
decile_table(test_e["target"], p_e)

# %%
joblib.dump({"lgb": m_lgb, "xgb": m_xgb, "catboost": m_cb, "features": F3, "cat_cols": cat_cols},
            "../models/session_start_blend.joblib")
joblib.dump({"model": m_early, "features": F3 + EARLY}, "../models/early_session_lgbm.joblib")
