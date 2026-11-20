# %% [markdown]
# # 05. Final model and the December test
#
# Chosen on November: feature set **E5** (no raw campaign / ad IDs), LightGBM with `num_leaves=127`.
# Now we retrain on **May–November** and score **December once**.

# %%
import warnings
warnings.filterwarnings("ignore")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

exec(open("04_experiments.py").read().split("# %%\nexperiments")[0])   # same feature lists and params

E5 = RAW_OTHER + GENERAL + PAST_CR + VISITOR + LANDING
CATS = [c for c in ALL_CATS if c in E5]

full = pd.read_parquet(f"{PROCESSED}/features.parquet")
fit_df = full[full["visit_dt"] < "2021-12-01"].copy()
test = full[full["visit_dt"] >= "2021-12-01"].copy()
for c in CATS:
    cats = pd.CategoricalDtype(fit_df[c].unique())
    fit_df[c] = fit_df[c].astype(cats)
    test[c] = test[c].astype(cats)

# %% [markdown]
# Number of trees: best iteration on November (128), scaled up ~20% because the training set is bigger.

# %%
params = {**PARAMS, "num_leaves": 127, "min_child_samples": 200, "n_estimators": 155}
model = lgb.LGBMClassifier(**params).fit(fit_df[E5], fit_df["target"])
proba = model.predict_proba(test[E5])[:, 1]

auc = roc_auc_score(test["target"], proba)
ap = average_precision_score(test["target"], proba)
print(f"December test — ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f} (random = {test['target'].mean():.4f})")

# %% [markdown]
# ## What the score means for the business
# Rank December sessions by predicted probability and split them into 10 equal groups (deciles).
# **Lift** = conversion rate in the group ÷ average conversion rate.

# %%
test["score"] = proba
test["decile"] = pd.qcut(test["score"].rank(method="first", ascending=False), 10, labels=range(1, 11))
dec = test.groupby("decile", observed=True)["target"].agg(sessions="size", conversions="sum", cr="mean")
dec["lift"] = (dec["cr"] / test["target"].mean()).round(2)
dec["cum_share_of_conversions"] = (dec["conversions"].cumsum() / dec["conversions"].sum() * 100).round(1)
dec["cr"] = (dec["cr"] * 100).round(2)
dec

# %% [markdown]
# ## Feature importance (gain)

# %%
imp = (pd.Series(model.booster_.feature_importance("gain"), index=E5)
         .sort_values(ascending=False))
(imp / imp.sum() * 100).round(1).head(15)

# %%
joblib.dump({"model": model, "features": E5, "categories": {c: fit_df[c].cat.categories for c in CATS}},
            "../models/lgbm_conversion.joblib")
test[["session_id", "visit_dt", "target", "score"]].to_parquet(f"{PROCESSED}/test_scores.parquet", index=False)
